# UPOCR (ICML 2024)

The official implementation of [UPOCR: Towards Unified Pixel-Level OCR Interface](https://arxiv.org/abs/2312.02694) (ICML 2024).
The UPOCR represents a first-of-its-kind simple-yet-effective generalist model for unified pixel-level OCR interface.
Through the unification of paradigms, architectures, and training strategies, UPOCR simultaneously excels in diverse pixel-level OCR tasks using a single unified model.
Below is the framework of UPOCR.

![UPOCR](figures/method.svg)

## Environment

We recommend using [Anaconda](https://www.anaconda.com/) to manage environments. Run the following commands to install dependencies.
```
conda create -n upocr python=3.9 -y
conda activate upocr
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
git clone https://github.com/shannanyinxiang/UPOCR.git
cd UPOCR
pip install -r requirements.txt
```

## Datasets

1. Download the SCUT-EnsText [[repo]](https://github.com/HCIILAB/SCUT-EnsText), TextSeg [[repo]](https://github.com/SHI-Labs/Rethinking-Text-Segmentation), and Tampered-IC13 [[repo]](https://github.com/wangyuxin87/Tampered-IC13) datasets. 
2. Preprocess the SCUT-EnsText dataset following [[link]](https://github.com/shannanyinxiang/ViTEraser?tab=readme-ov-file#1-text-removal-dataset).
3. Arrange the datasets according to the file structure below.
```
data
├─TamperedTextDetection
│  └─Tampered-IC13
│     ├─test_gt
│     ├─test_img
│     ├─train_gt  
│     └─train_img
├─TextRemoval
│  └─SCUT-EnsText
│     ├─train
│     │  ├─image
│     │  ├─label
│     │  └─mask
│     └─test
│        ├─image
│        ├─label
│        └─mask
└─TextSegmentation
   └─TextSeg
      ├─image
      ├─semantic_label
      └─split.json
```

## Inference

- Download the UPOCR weights at [[link]](https://pan.baidu.com/s/1DrCOOVGykLiIC_xxRqkPOg?pwd=mdim).
- Run the following command to perform model inference on the TextSeg dataset.
```
dataset=textseg #  or scut-enstext or tampered-ic13 
output_dir=./output/upocr-infer/

mkdir ${output_dir}

CUDA_VISIBLE_DEVICES=0 \
torchrun \
        --master_port=3140 \
        --nproc_per_node=1 \
        main.py \
        --output_dir ${output_dir} \
        --data_cfg_paths data_configs/train/scut-enstext.yaml data_configs/train/tampered-ic13.yaml data_configs/train/textseg.yaml \
        --eval true \
        --resume pretrained/upocr.pth \
        --eval_data_cfg_path data_configs/eval/${dataset}.yaml \
        --visualize true
```
Change the `dataset` variable to `scut-enstext` or `tampered-ic13` to run inference on the SCUT-EnsText or Tampered-IC13 datasets, respectively.

- For the text removal task, run the following command to calculate image-eval metrics. 
For the other two tasks, the metrics will be automatically calculated at the above step.
```
python -u eval/text_removal/evaluation.py \
    --gt_path data/TextErase/SCUT-ENS/test/label/ \
    --target_path output/upocr-infer/SCUT-EnsText

python -m pytorch_fid \
    data/TextErase/SCUT-ENS/test/label/ \
    output/upocr-infer/SCUT-EnsText \
    --device cuda:0
```

## Training

- Download the pre-training weights for UPOCR at [[link]](https://pan.baidu.com/s/1jLp0YwRcSJqqhNPJHnCBUQ?pwd=3bqa).
- Run the following command for model training.
```
output_dir=./output/upocr-train/
log_path=${output_dir}log_train.txt

mkdir 'output'
mkdir ${output_dir}

CUDA_VISIBLE_DEVICES=0,1 \
torchrun \
        --master_port=3140 \
        --nproc_per_node=2 \
        main.py \
        --output_dir ${output_dir} \
        --data_cfg_paths data_configs/train/scut-enstext.yaml data_configs/train/tampered-ic13.yaml data_configs/train/textseg.yaml \
        --pretrained_model pretrained/pretraining_weights.pth \
        --amp true | tee -a ${log_path}
```

## Citation
```
@inproceedings{peng2024upocr,
  title={{UPOCR}: Towards Unified Pixel-Level {OCR} Interface},
  author={Peng, Dezhi and Yang, Zhenhua and Zhang, Jiaxin and Liu, Chongyu and Shi, Yongxin and Ding, Kai and Guo, Fengjun and Jin, Lianwen},
  booktitle={International Conference on Machine Learning},
  year={2024},
}
```

## Copyright
This repository can only be used for non-commercial research purpose.

For commercial use, please contact Prof. Lianwen Jin (eelwjin@scut.edu.cn).

Copyright 2024, [Deep Learning and Vision Computing Lab](http://www.dlvc-lab.net), South China University of Technology. 
