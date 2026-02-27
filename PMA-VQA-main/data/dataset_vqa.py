import os
import sys
import torch.utils.data as data
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import random
import json

from bert.tokenization_bert import BertTokenizer

from args import get_parser


class VQADataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 split='train',
                 eval_mode=False):

        self.image_transforms = image_transforms
        self.split = split
        self.args = args

        # 动态加载答案类型
        self.answers = self._load_answers()
        self.answer2id = {a: i for i, a in enumerate(self.answers)}
        self.id2answer = {i: a for a, i in self.answer2id.items()}

        self.max_tokens = 30
        self.eval_mode = eval_mode

        # 加载数据
        self.data = self._load_data()

        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_tokenizer)

    def _load_answers(self):
        """动态加载答案类型"""
        if hasattr(self.args, 'answers_file') and self.args.answers_file:
            answers_path = self.args.answers_file
            if os.path.exists(answers_path):
                try:
                    with open(answers_path, 'r', encoding='utf-8') as f:
                        answers_config = json.load(f)
                    if isinstance(answers_config, list):
                        print(f"Loaded {len(answers_config)} answer types from {answers_path}")
                        return answers_config
                    elif isinstance(answers_config, dict) and 'answers' in answers_config:
                        print(f"Loaded {len(answers_config['answers'])} answer types from {answers_path}")
                        return answers_config['answers']
                    else:
                        print(f"Invalid answers file format in {answers_path}")
                        raise ValueError("Invalid answers file format")
                except Exception as e:
                    print(f"Error loading answers file {answers_path}: {e}")
                    raise ValueError(f"Failed to load answers file: {e}")

        # 如果没有指定答案文件，抛出错误
        raise ValueError("answers_file must be specified in args")

    def _load_data(self):
        # 构建JSON文件路径
        if self.split == 'train' and hasattr(self.args, 'train_json'):
            json_filename = self.args.train_json
        elif self.split == 'val' and hasattr(self.args, 'val_json'):
            json_filename = self.args.val_json
        else:
            json_filename = f'{self.split}.json'

        json_path = json_filename

        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found, returning empty dataset")
            return []

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            # 转换数据格式
            processed_data = []
            for item in raw_data:
                # 构建完整的图像路径
                image_path = os.path.join(self.args.images_dir, item['image_name'])

                processed_item = {
                    'image_path': image_path,
                    'question': item['question'],
                    'answer': item['answer'],
                    'qid': item.get('qid', 0),
                    'question_type': item.get('question_type', 'unknown')
                }
                processed_data.append(processed_item)

            print(f"Loaded {len(processed_data)} samples from {json_path}")
            return processed_data

        except Exception as e:
            print(f"Error loading data from {json_path}: {e}")
            return []

    def _bin_count_lr(self, n_str):
        """LR数据集的数字答案分箱处理（计数问题）"""
        try:
            n = int(n_str)
            if n == 0:
                return "0"
            elif 1 <= n <= 10:
                return "between 1 and 10"
            elif 11 <= n <= 100:
                return "between 11 and 100"
            elif 101 <= n <= 1000:
                return "between 101 and 1000"
            elif n > 1000:
                return "more than 1000"
            else:
                return None
        except (ValueError, TypeError):
            return None

    def _bin_area_hr(self, n_str):
        try:
            if 'm2' in n_str:
                num_str = n_str.replace('m2', '').strip()
                if num_str.isdigit():
                    num = int(num_str)

                    if num == 0:
                        return "0m2"
                    elif 1 <= num <= 10:
                        return "between 1m2 and 10m2"
                    elif 11 <= num <= 100:
                        return "between 11m2 and 100m2"
                    elif 101 <= num <= 1000:
                        return "between 101m2 and 1000m2"
                    elif num > 1000:
                        return "more than 1000m2"
                    else:
                        return None
            else:
                return None

        except (ValueError, TypeError):
            return None

    def normalize_answer(self, ans):
        """规范化答案到预定义的答案类型之一"""
        if ans is None:
            return None

        import re
        import unicodedata

        def _clean(s):
            s = str(s).strip().lower()
            s = re.sub(r'[^\w\s]', ' ', s)
            s = re.sub(r'\s+', ' ', s)
            s = s.strip()
            return s

        def _to_int_simple(s):
            m = re.findall(r'\d+', s)
            return m[0] if m else None

        s = _clean(ans)

        if hasattr(self.args, 'dataset'):
            if self.args.dataset == 'LR':
                n_str = _to_int_simple(s)
                if n_str is not None:
                    b = self._bin_count_lr(n_str)
                    if b: return b

            elif self.args.dataset == 'HR':
                if 'm2' in s:
                    b = self._bin_area_hr(s)
                    if b: return b

        if s in self.answers:
            return s

        for answer in self.answers:
            if s in answer.lower() or answer.lower() in s:
                return answer

        return None

    def get_classes(self):
        return self.answers

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        img = Image.open(item['image_path']).convert("RGB")

        question = item['question']

        raw_answer = item['answer']
        norm_answer = self.normalize_answer(raw_answer)

        if norm_answer is None:
            print(f"Warning: Could not normalize answer '{raw_answer}' for question: {question}")
            return self.__getitem__((index + 1) % len(self.data))

        answer_id = self.answer2id[norm_answer]

        if self.image_transforms is not None:
            img = self.image_transforms(img)

        input_ids = self.tokenizer.encode(text=question, add_special_tokens=True)
        input_ids = input_ids[:self.max_tokens]

        padded_input_ids = [0] * self.max_tokens
        padded_input_ids[:len(input_ids)] = input_ids

        attention_mask = [0] * self.max_tokens
        attention_mask[:len(input_ids)] = [1] * len(input_ids)

        input_ids = torch.tensor(padded_input_ids)
        attention_mask = torch.tensor(attention_mask)
        answer_id = torch.tensor(answer_id)

        question_type = item['question_type']

        return img, answer_id, input_ids, attention_mask, question_type
