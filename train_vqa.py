import datetime
import os
import time
import sys

import torch
import torch.utils.data
from torch import nn

from functools import reduce
import operator
from bert.modeling_bert import BertModel

import torchvision
from lib import model_builder

from torchvision import transforms as T
from PIL import Image

import utils
import numpy as np

import torch.nn.functional as F

import gc
from collections import OrderedDict


def get_dataset(image_set, transform, args):
    from data.dataset_vqa import VQADataset
    ds = VQADataset(args,
                    split=image_set,
                    image_transforms=transform
                    )
    num_classes = len(ds.answers)

    return ds, num_classes


def accuracy(pred, target):
    pred = pred.argmax(1)
    correct = (pred == target).float().sum()
    total = target.size(0)
    return correct / total


def get_transform(args):
    return T.Compose([
        T.Resize((args.img_size, args.img_size), interpolation=Image.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def criterion(input, target):
    return nn.functional.cross_entropy(input, target)


def setup_logging(args):
    if not utils.is_main_process():
        return None

    os.makedirs(args.output_dir, exist_ok=True)
    if getattr(args, "log_file", ""):
        log_file = args.log_file
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(args.output_dir, f"train_{args.model_id}_{timestamp}.log")

    class _Tee(object):
        def __init__(self, stream, path):
            self.stream = stream
            self.log = open(path, "a", buffering=1)

        def write(self, data):
            self.stream.write(data)
            self.log.write(data)

        def flush(self):
            self.stream.flush()
            self.log.flush()

    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = sys.stdout
    print(f"[Logging] save log to {log_file}", force=True)
    return log_file


def evaluate(model, data_loader, bert_model):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_its = 0
    acc_sum = 0

    question_type_stats = {}

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            image, target, sentences, attentions, question_types = data
            image, target, sentences, attentions = image.cuda(non_blocking=True), \
                target.cuda(non_blocking=True), \
                sentences.cuda(non_blocking=True), \
                attentions.cuda(non_blocking=True)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            if bert_model is not None:
                last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
                embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
                attentions = attentions.unsqueeze(dim=-1)  # (B, N_l, 1)
                output = model(image, embedding, l_mask=attentions)
            else:
                output = model(image, sentences, l_mask=attentions)

            acc = accuracy(output, target)
            acc_sum += acc.item()

            batch_size = target.size(0)
            for i in range(batch_size):
                question_type = question_types[i]

                if question_type not in question_type_stats:
                    question_type_stats[question_type] = {
                        'correct': 0,
                        'total': 0
                    }

                pred_i = output[i].argmax(0)
                if pred_i == target[i]:
                    question_type_stats[question_type]['correct'] += 1
                question_type_stats[question_type]['total'] += 1

        avg_acc = acc_sum / total_its

    print('Final results:')
    print('Overall Accuracy: %.2f%%' % (avg_acc * 100.))

    print('\nPer-question-type Accuracy:')
    total_question_types = 0
    total_question_type_acc = 0.0

    for q_type, stats in question_type_stats.items():
        q_acc = (stats['correct'] / stats['total']) * 100.0
        total_question_types += 1
        total_question_type_acc += q_acc

        print(f'  {q_type}: {q_acc:.2f}% ({stats["correct"]}/{stats["total"]})')

    mean_question_type_acc = total_question_type_acc / total_question_types if total_question_types > 0 else 0.0
    print(f'\nMean Question Type Accuracy: {mean_question_type_acc:.2f}%')

    return 100 * avg_acc, mean_question_type_acc, question_type_stats


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, bert_model):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

    for data in metric_logger.log_every(data_loader, print_freq, header):
        total_its += 1
        image, target, sentences, attentions, question_types = data
        image, target, sentences, attentions = image.cuda(non_blocking=True), \
            target.cuda(non_blocking=True), \
            sentences.cuda(non_blocking=True), \
            attentions.cuda(non_blocking=True)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        if bert_model is not None:
            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]  # (B, N_l, 768)
            embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
            output = model(image, embedding, l_mask=attentions)
        else:
            output = model(image, sentences, l_mask=attentions)

        loss = criterion(output, target)
        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        torch.cuda.synchronize()
        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        del image, target, sentences, attentions, loss, output, data
        if bert_model is not None:
            del last_hidden_states, embedding

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main(args):
    setup_logging(args)

    dataset, num_classes = get_dataset("train", get_transform(args=args), args=args)
    dataset_test, _ = get_dataset("val", get_transform(args=args), args=args)

    # batch sampler
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                                    shuffle=True)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=8, sampler=test_sampler, num_workers=args.workers)

    args.num_answers = num_classes
    print(f"Number of answer classes: {num_classes}")

    # model initialization
    print(args.model)
    model = model_builder.__dict__[args.model](pretrained=args.pretrained_swin_weights,
                                              args=args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    single_model = model.module


    model_class = BertModel
    bert_model = model_class.from_pretrained(args.ck_bert)
    bert_model.pooler = None  # a work-around for a bug in Transformers = 3.0.2 that appears for DistributedDataParallel
    bert_model.cuda()
    bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)
    bert_model = torch.nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank])
    single_bert_model = bert_model.module

    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])
        single_bert_model.load_state_dict(checkpoint['bert_model'])

    # parameters to optimize
    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in single_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    params_to_optimize = [
        {'params': backbone_no_decay, 'weight_decay': 0.0},
        {'params': backbone_decay},
        {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
        # the following are the parameters of bert
        {"params": reduce(operator.concat,
                          [[p for p in single_bert_model.encoder.layer[i].parameters()
                            if p.requires_grad] for i in range(10)])},
    ]

    # optimizer
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    # housekeeping
    start_time = time.time()
    iterations = 0
    best_acc = -0.1

    # resume training (optimizer, lr scheduler, and the epoch)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
    else:
        resume_epoch = -999

    # training loops
    for epoch in range(max(0, resume_epoch + 1), args.epochs):
        data_loader.sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, args.print_freq,
                        iterations, bert_model)
        overall_acc, mean_question_type_acc, question_type_stats = evaluate(model, data_loader_test, bert_model)

        print('Overall Accuracy: %.2f%%' % overall_acc)
        print('Mean Question Type Accuracy: %.2f%%' % mean_question_type_acc)

        # 保存最佳模型（基于总体准确率）
        save_checkpoint = (best_acc < overall_acc)
        if save_checkpoint:
            print('Better epoch: {}\n'.format(epoch))
            if single_bert_model is not None:
                dict_to_save = {'model': single_model.state_dict(), 'bert_model': single_bert_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict()}
            else:
                dict_to_save = {'model': single_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict()}

            utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                            'model_best_{}.pth'.format(args.model_id)))
            best_acc = overall_acc

    # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from args import get_parser

    parser = get_parser()
    args = parser.parse_args()
    # set up distributed learning
    utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
