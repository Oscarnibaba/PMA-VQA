# PMA-VQA
PMA-VQA: Progressive Multi-Scale Feature Fusion with Spatial Adaptive Attention for Remote Sensing Visual Question Answering

## Directory Structure

```text
PMA-VQA/
├── bert/                    # BERT language encoder [based on HuggingFace Transformers v3.0.2](https://huggingface.co/transformers/v3.0.2/quicktour.html)
│   ├── ...                  
├── lib/
│   ├── backbone.py          # Implementation of the Swin Transformer backbone network
│   ├── segmentation.py      # Functions for building the VQA model architecture
│   ├── _utils.py            # Wrapper utilities for the VQA model (training & inference helpers)
│   └── vqa_head.py          # VQA classification head for answer prediction
├── data/
│   ├── dataset_vqa.py       # Definition of the VQA dataset class
├── train_vqa.py             # Main script for training the VQA model
├── test_vqa.py              # Main script for testing the VQA model
├── args.py                  # Configuration of training arguments and hyperparameters
├── transforms.py            # Image preprocessing and data augmentation functions
└── utils.py                 # General utility functions
```

## Setting Up

### Preliminaries

The code has been verified to work with PyTorch v1.7.1 and Python 3.7.
1. Clone this repository.
2. Change directory to root of this repository.

### Package Dependencies

- Create a new Conda environment with Python 3.7 then activate it:

```shell
conda create -n pma-vqa python==3.7
conda activate pma-vqa
```

- Install PyTorch v1.7.1 with a CUDA version that works on your cluster/machine (CUDA 10.2 is used in this example):

```shell
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
```

- Install the packages in `requirements.txt` via `pip`:

```shell
pip install -r requirements.txt
```

### Datasets
- Download the [RSVQA](https://github.com/syvlo/RSVQA)
- Data Preprocessing: Convert your original VQA annotations into this format by:
```json
{
    "qid": 47218,
    "image_name": "472.tif",
    "question_type": "rural_urban",
    "question": "Is it a rural or an urban area?",
    "answer": "rural"
}
```

### Weights for Training

-  Create the `./pretrained_weights` directory where we will be storing the weights.

```shell
mkdir ./pretrained_weights
```

-  The original Swin Transformer. Download [pre-trained classification weights of
the Swin Transformer](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth), `swin_base_patch4_window12_384_22k.pth`, into `./pretrained_weights`.
These weights are needed in training to initialize the model.

- Create the `./checkpoints` directory where the program will save the weights during training.

```shell
mkdir ./checkpoints
```


- The following script starts training on the LR dataset :
```shell
python -m torch.distributed.launch \
          --nproc_per_node 4  \
            train_vqa.py 
          --dataset LR \
          --images_dir /data/dateaset/LR/Images_LR \
          --train_json /data/dateaset/LR/LR_Train.json \
          --val_json /data/dateaset/LR/LR_Val.json \
          --answers_file /data/dateaset/LR/answers_list_test.json \
          --batch-size 16 \
          --lr 0.00005 \
          --wd 1e-2 \
          --swin_type base \
          --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
          --ck_bert ./pretrained_ckp/bert-base-uncased/ \
          --bert_tokenizer ./pretrained_ckp/bert-base-uncased/ \
          --epochs 10 \
          --img_size 384 \
          --use_multi_scale \
          --output-dir ./checkpoint/
```
- The trained checkpoints for the LR and HR datasets have been uploaded, which can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1K1q6irQL1xPZtqv5O4yYsQ). Extraction code: a47u.

###  Test
```shell
python test_vqa.py \
    --test_json  /data/datasets/LR/LR_Test.json \
    --images_dir /data/datasets/LR/Images_LR \
    --answers_file /data//datasets/LR/answers_list_test.json \
    --bert_tokenizer ./pretrained_ckp/bert-base-uncased/ \
    --ck_bert ./pretrained_ckp/bert-base-uncased/ \
    --dataset LR \
    --img_size 384 \
    --swin_type base \
    --window12 \
    --use_multi_scale \
    --checkpoint ./checkpoint/LR/model_best_LR.pth \
    --output_test ./checkpoint/LR/test_results.json \
    --batch-size 16
```
