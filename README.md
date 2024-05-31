# NVIB FS Classfication (Vision)
---







## Table of Content
* [Prerequisites](#prerequisites)
* [Datasets](#datasets)
    * [CIFAR-FS](#cifar-fs-and-mini-imagenet)
    * [Meta-Dataset](#meta-dataset)
* [Meta-training](#meta-training)
    * [On CIFAR-FS](#on-cifar-fs-and-mini-imagenet)
    * [On Meta-Dataset](#on-meta-dataset-with-imagenet-only)
* [Meta-testing](#meta-testing)
    * [CIFAR-FS (In-Domain)](#for-datasets-without-domain-shift)
    * [Meta-Dataset (Out-Of-Domain)](#fine-tuning-on-meta-test-tasks)





## Prerequisites
```
pip install -r requirements.txt
```
The code was tested with Python 3.8.1 and Pytorch >= 1.7.0.


## Datasets


### CIFAR-FS 
```
cd scripts
sh download_cifarfs.sh
```
To use these two datasets, set `--dataset cifar_fs` or `--dataset mini_imagenet`.

### Meta-Dataset
Implementation of meta-dataset is based on [pmf_cvpr22](https://github.com/hushell/pmf_cvpr22).
The dataset has 10 domains, 4000+ classes. Episodes are formed in various-way-various-shot fashion, where an episode can have 900+ images.
The images are stored class-wise in h5 files (converted from the origianl tfrecords, one for each class).
To train and test on this dataset, set `--dataset meta_dataset` and `--data_path /path/to/meta_dataset`.

To download the h5 files, 
```
git clone https://huggingface.co/datasets/hushell/meta_dataset_h5
```

## Pre-training
The  architecture we used is DeiT-Small (deit_small_patch16_224). The pretrained weight will be loaded automatically.

## Default NVIB parameters

* `--nvib`: Enable NVIB layers.
* `--nvib_layers 0 1 2 3 4 5`: Specify layers with NVIB inserted.
* `--delta 1`:  Conditional prior for dirichlet KL divergence
* `--alpha_tau 0`: Initialisation for dirichlet projections
* `--stdev_tau 0 `: Initialisation for gaussian variance projections
* `--lambda_klg 0.01 ` and `--lambda_kld 0.01`: Weight for the KL divergence between the Gaussian/Dirichlet  prior and the posterior

## Meta-Training

### On CIFAR-FS
It is recommended to run on one GPU.
We use `args.nSupport` to set the number of shots. The 5-way-5-shot training command of CIFAR-FS writes as
```
python -m torch.distributed.launch --nproc_per_node=1   --use_env  main.py   --output outputs/your_experiment_name  --nvib --nvib_layers 0 1 2 3 4 5  --delta 1 --alpha_tau 0  --stdev_tau 0 --lambda_klg 0.01 --lambda_kld 0.01    --dataset cifar_fs --epoch 100 --lr 1e-4 --arch deit_small_patch16 --dist-eval --nSupport 5 --fp16   
```

### On Meta-Dataset (Trained on ImageNet only)
It is recommended to run on four GPU.
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --output outputs/your_experiment_name --nvib --nvib_layers 0 1 2 3 4 5  --delta 1  --alpha_tau -3 --stdev_tau 0 --lambda_klg 0.001 --lambda_kld 0.001  --dataset meta_dataset --data-path /path/to/meta-dataset/ --num_workers 4  --base_sources ilsvrc_2012  --test_sources traffic_sign mscoco  omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower --epoch 100 --lr 5e-5 --arch deit_small_patch16 --dist-eval  --fp16
```
Set   `--base_sources ilsvrc_2012` to enable training on ImageNet.


## Meta-Testing
Set ```--resume``` as your trained model's path.
### For datasets without domain shift (CIFAR-FS)
```
python -m torch.distributed.launch --nproc_per_node=1   --use_env  main.py  --resume outputs/your_experiment_name/best.pth  --output outputs/your_experiment_name  --nvib --nvib_layers 0 1 2 3 4 5  --delta 1 --alpha_tau 0  --stdev_tau 0 --lambda_klg 0.01 --lambda_kld 0.01   --dataset cifar_fs --epoch 100 --lr 1e-4 --arch deit_small_patch16 --eval --nSupport 5 --fp16  
```
### Fine-tuning on meta-test tasks

A meta-testing command example for Meta-Dataset is: 
``` 
python -m torch.distributed.launch --nproc_per_node=1 --use_env test_meta_dataset.py --resume outputs/your_experiment_name/best.pth  --output outputs/your_experiment_name  --data-path /path/to/meta_dataset/ --nvib --nvib_layers 0 1 2 3 4 5 -delta 1 --alpha_tau -3 --stdev_tau 0 --lambda_klg 0.001 --lambda_kld 0.001  --dataset meta_dataset --arch deit_small_patch16 --base_sources ilsvrc_2012 --test_sources traffic_sign mscoco  omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower  --dist-eval 
``` 



