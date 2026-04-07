import os
import json
import datetime
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from torchvision import transforms as T

from bert.modeling_bert import BertModel
from lib import model_builder
from data.dataset_vqa import VQADataset
from args import get_parser


def get_transform(img_size):
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=Image.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model_and_bert(args, checkpoint_path, device):
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    saved_args = checkpoint.get('args', None)
    if saved_args:
        args.num_answers = getattr(saved_args, 'num_answers', args.num_answers)
        print(f"Using num_answers from checkpoint: {args.num_answers}")

    model = model_builder.__dict__[args.model](pretrained='', args=args)

    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    bert_model = BertModel.from_pretrained(args.ck_bert)
    bert_model.pooler = None

    if 'bert_model' in checkpoint:
        bert_model.load_state_dict(checkpoint['bert_model'])

    bert_model = bert_model.to(device)
    bert_model.eval()

    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    return model, bert_model


def evaluate(model, bert_model, data_loader, device, id2answer):
    """评估模型"""
    model.eval()
    bert_model.eval()

    all_results = []
    correct = 0
    total = 0

    question_type_stats = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            images, answer_ids, input_ids, attention_mask, question_types = batch

            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            answer_ids = answer_ids.to(device)

            last_hidden_states = bert_model(input_ids, attention_mask=attention_mask)[0]
            embedding = last_hidden_states.permute(0, 2, 1)
            l_mask = attention_mask.unsqueeze(dim=-1)

            outputs = model(images, embedding, l_mask=l_mask)
            predictions = outputs.argmax(dim=1)

            batch_size = images.size(0)
            for i in range(batch_size):
                pred_id = predictions[i].item()
                gt_id = answer_ids[i].item()
                q_type = question_types[i]

                is_correct = (pred_id == gt_id)
                if is_correct:
                    correct += 1
                total += 1

                if q_type not in question_type_stats:
                    question_type_stats[q_type] = {'correct': 0, 'total': 0}
                question_type_stats[q_type]['total'] += 1
                if is_correct:
                    question_type_stats[q_type]['correct'] += 1

                all_results.append({
                    'ground_truth': id2answer.get(gt_id, str(gt_id)),
                    'prediction': id2answer.get(pred_id, str(pred_id)),
                    'correct': is_correct,
                    'question_type': q_type
                })

    overall_acc = correct / total if total > 0 else 0
    return overall_acc, question_type_stats, all_results


def main():
    parser = get_parser()
    args = parser.parse_args()

    if not args.checkpoint:
        print("Error: --checkpoint is required!")
        return
    if not args.test_json:
        print("Error: --test_json is required!")
        return

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} not found!")
        return

    if not os.path.exists(args.test_json):
        print(f"Error: Test JSON file {args.test_json} not found!")
        return

    args.val_json = args.test_json

    print("Loading test dataset...")
    transform = get_transform(args.img_size)
    test_dataset = VQADataset(args, image_transforms=transform, split='val', eval_mode=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    args.num_answers = len(test_dataset.answers)
    print(f"Number of answer classes: {args.num_answers}")

    print("Loading model...")
    model, bert_model = load_model_and_bert(args, args.checkpoint, device)

    print("Starting evaluation...")
    overall_acc, question_type_stats, all_results = evaluate(
        model, bert_model, test_loader, device, test_dataset.id2answer
    )

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy: {overall_acc * 100:.2f}%")

    print("\nPer-question-type Accuracy:")
    total_q_acc = 0
    for q_type, stats in sorted(question_type_stats.items()):
        q_acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        total_q_acc += q_acc
        print(f"  {q_type}: {q_acc:.2f}% ({stats['correct']}/{stats['total']})")

    mean_q_acc = total_q_acc / len(question_type_stats) if question_type_stats else 0
    print(f"\nMean Question Type Accuracy: {mean_q_acc:.2f}%")

    results_summary = {
        'checkpoint': args.checkpoint,
        'test_json': args.test_json,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'overall_accuracy': overall_acc,
        'mean_question_type_accuracy': mean_q_acc / 100,
        'question_type_stats': {
            q_type: {
                'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
                'correct': stats['correct'],
                'total': stats['total']
            }
            for q_type, stats in question_type_stats.items()
        },
        'total_samples': len(all_results),
        'correct_samples': sum(1 for r in all_results if r['correct']),
        'detailed_results': all_results
    }

    output_dir = os.path.dirname(args.output_test)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_test, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {args.output_test}")
    print("=" * 60)


if __name__ == '__main__':
    main()
