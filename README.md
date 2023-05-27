# SMMix
This a Pytorch implementation of our paper "SMMix: Self-Motivated Image Mixing for Vision Transformers"


## Requirements
- python 3.8.0
- pytorch 1.7.1
- torchvision 0.8.2


## Data Preparation
- The ImageNet dataset should be prepared as follows:
```
ImageNet
├── train
│   ├── folder 1 (class 1)
│   ├── folder 2 (class 2)
│   ├── ...
├── val
│   ├── folder 1 (class 1)
│   ├── folder 2 (class 2)
│   ├── ...

```

## Pre-trained Models

|Model|Top-1 Accuracy|Dowmload|
|-----|:-------:|---------|
|DeiT-T| 73.6 |[model](https://drive.google.com/file/d/1cclVfL_dCs7uQIdEvUoTsdMnc2gN7tZD/view?usp=sharing) & [log](https://drive.google.com/file/d/13VcsQeX1X6ONEI4pZp9biPtmOeXl9EsV/view?usp=sharing)|
|DeiT-S| 81.1 |[model](https://drive.google.com/file/d/1FYRglSq7EFVDVAE1sOZ81bBm_fZj_9bz/view?usp=share_link) & [log](https://drive.google.com/file/d/1rd3_HzHyCAhocLbO78tzAaXsyJUzQJGy/view?usp=share_link)|
|PVT-T | 76.4 |[model](https://drive.google.com/file/d/11ULLrgyPbeBr3TZXEh7xIb3HXjozCKFL/view?usp=sharing) & [log](https://drive.google.com/file/d/1e7m8K57fWcPawAEtUtkTDaTlZXJbU1gz/view?usp=sharing)|
|PVT-S | 81.0 |[model](https://drive.google.com/file/d/18QH-IEOI6KYpbST0xMyjJBTJeKFxtw2U/view?usp=sharing) & [log](https://drive.google.com/file/d/1yKgfi1dpb0puFhZy-pZJF-1vBWHM_scs/view?usp=sharing)|
|PVT-M | 82.2 |[model](https://drive.google.com/file/d/1AEN7iiIYABmaHCkK9ds9A-owEuBIX2mU/view?usp=sharing) & [log](https://drive.google.com/file/d/1hgycht1Szor9aUePbCyOyoDyId9yBkNf/view?usp=sharing)|
|PVT-L | 82.7 |[model](https://drive.google.com/file/d/1IG-XONNBfv-Rg5ETZfTfVa2i5qk11lNr/view?usp=sharing) & [log](https://drive.google.com/file/d/1aojFiCSv_eZtYJBaCuRwCTZFQqOT-sYu/view?usp=sharing)|

## Evaluation
```
./script/eval.sh --data-path DATASET_PATH --model MODEL_NAME --resume CHECKPOINT_PATH
```
examples:
```
./script/eval.sh --data-path /media/DATASET/ImageNet --model pvt_small --resume ./checkpoints/pvt_small_smmix.pth
```
```
./script/eval.sh --data-path /media/DATASET/ImageNet --model vit_deit_small_patch16_224 --resume ./checkpoints/deit_small_smmix.pth
```

## Training
```
./script/train.sh --data-path DATASET_PATH --model MODEL_NAME --output_dir LOG_PATH --batch_size 256
```
examples:
```
./script/train.sh --data-path /media/DATASET/ImageNet --model pvt_small --output_dir ./log/pvt_small_smmix --batch_size 256
```
```
./script/train.sh --data-path /media/DATASET/ImageNet --model vit_deit_small_patch16_224 --output_dir ./log/deit_small_smmix --batch_size 256
```
