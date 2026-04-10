# PMA-VQA
PMA-VQA: Progressive Multi-Scale Feature Fusion with Spatial Adaptive Attention for Remote Sensing Visual Question Answering

## Directory Structure
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

## Setting Up

### Preliminaries

The code has been verified to work with PyTorch v1.7.1 and Python 3.7.
1. Clone this repository.
2. Change directory to root of this repository.

### Package Dependencies

1. Create a new Conda environment with Python 3.7 then activate it:

```shell
conda create -n lavt python==3.7
conda activate pma-vqa
```

2. Install PyTorch v1.7.1 with a CUDA version that works on your cluster/machine (CUDA 10.2 is used in this example):

```shell
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
```

3. Install the packages in `requirements.txt` via `pip`:

```shell
pip install -r requirements.txt
```

### Datasets

#### Image



### Weights for Training

1.  Create the `./pretrained_weights` directory where we will be storing the weights.

```shell
mkdir ./pretrained_weights
```

2.  The original Swin Transformer. Download [pre-trained classification weights of
the Swin Transformer](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth), `swin_base_patch4_window12_384_22k.pth`, into `./pretrained_weights`.
These weights are needed in training to initialize the model.

3. Create the `./checkpoints` directory where the program will save the weights during training.

```shell
mkdir ./checkpoints
```
