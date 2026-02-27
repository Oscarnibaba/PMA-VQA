import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='LAVT VQA training and testing')
    parser.add_argument('--amsgrad', action='store_true',
                        help='if true, set amsgrad to True in an Adam or AdamW optimizer.')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--bert_tokenizer', default='bert-base-uncased', help='BERT tokenizer')
    parser.add_argument('--ck_bert', default='bert-base-uncased', help='pre-trained BERT weights')
    parser.add_argument('--dataset', default='vqa', help='vqa dataset name')
    parser.add_argument('--images_dir', default='images/', help='images subdirectory name')
    parser.add_argument('--train_json', default='train.json', help='training JSON file name')
    parser.add_argument('--val_json', default='val.json', help='validation JSON file name')
    parser.add_argument('--answers_file', default='', help='path to answers configuration file (JSON format)')
    parser.add_argument('--ddp_trained_weights', action='store_true',
                        help='Only needs specified when testing,'
                             'whether the weights to be loaded are from a DDP-trained model')
    parser.add_argument('--device', default='cuda:0', help='device')  # only used when testing on a single machine
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--fusion_drop', default=0.0, type=float, help='dropout rate for SAAMs')
    parser.add_argument('--img_size', default=384, type=int, help='input image size')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--lr', default=0.00005, type=float, help='the initial learning rate')
    parser.add_argument('--mha', default='', help='If specified, should be in the format of a-b-c-d, e.g., 4-4-4-4,'
                                                  'where a, b, c, and d refer to the numbers of heads in stage-1,'
                                                  'stage-2, stage-3, and stage-4 SAAMs')
    parser.add_argument('--model', default='lavt_vqa', help='model: lavt_vqa (only option available)')
    parser.add_argument('--model_id', default='lavt_vqa', help='name to identify the model')
    parser.add_argument('--num_answers', default=9, type=int, help='number of answer classes')
    parser.add_argument('--use_multi_scale', action='store_true', help='use multi-scale feature fusion')
    parser.add_argument('--output-dir', default='./checkpoints/', help='path where to save checkpoint weights')
    parser.add_argument('--pin_mem', action='store_true',
                        help='If true, pin memory when using the data loader.')
    parser.add_argument('--pretrained_swin_weights', default='',
                        help='path to pre-trained Swin backbone weights')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--split', default='test', help='only used when testing')
    parser.add_argument('--swin_type', default='base',
                        help='tiny, small, base, or large variants of the Swin Transformer')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, metavar='W', help='weight decay',
                        dest='weight_decay')
    parser.add_argument('--window12', action='store_true',
                        help='only needs specified when testing,'
                             'when training, window size is inferred from pre-trained weights file name'
                             '(containing \'window12\'). Initialize Swin with window size 12 instead of the default 7.')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--log-file', default='',
                        help='路径可选：保存训练日志的文件（默认保存在output_dir下，主进程写入）')
    parser.add_argument('--checkpoint', type=str, default='', help='path to checkpoint file for testing')
    parser.add_argument('--test_json', type=str, default='', help='path to test JSON file')
    parser.add_argument('--output_test', type=str, default='test_results.json', help='output file for test results')
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()
