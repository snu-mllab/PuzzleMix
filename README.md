# Puzzle Mix: Exploiting Saliency and Local Statistics for Optimal Mixup

This is the code for the paper "Puzzle Mix: Exploiting Saliency and Local Statistics for Optimal Mixup" accepted at ICML'20 ([paper](https://arxiv.org/abs/2009.06962), [talk](https://icml.cc/virtual/2020/paper/6827), [blog](https://mllab.snu.ac.kr/kim-ICML2020.html)). Some parts of the codes are borrowed from manifold mixup ([link](https://github.com/vikasverma1077/manifold_mixup/tree/master/supervised)).

![Puzzle Mix image samples](figures/image_sample.png)

## Citing this Work 
```
@inproceedings{kimICML20,
    title= {Puzzle Mix: Exploiting Saliency and Local Statistics for Optimal Mixup},
    author = {Kim, Jang-Hyun and Choo, Wonho and Song, Hyun Oh},
    booktitle = {International Conference on Machine Learning (ICML)},
    year = {2020}
}
```

## Updates 
- (21.01.04) ImageNet training for 300 epochs is conducted! (Top-1 accuracy: **78.76%**, details are at [```./imagenet```](https://github.com/snu-mllab/PuzzleMix/tree/master/imagenet)).   
- (20.12.01/ torch 1.7) We built a **multi-processing** code for graph-cut, which runs on CPUs. As a result, the Puzzle Mix implementation (50s/epoch) is only slower about 1.5 times than Vanilla training (34s/epoch) on CIFAR-100, PreActResNet-18. 
To use the multi-processing, just simply add `--mp [n_procs]` in the command. 

## Requirements
This code has been tested with  
python 3.6.8  
pytorch 1.1.0  
torchvision 0.3.0  
gco-wrapper (https://github.com/Borda/pyGCO)

matplotlib 2.1.0  
numpy 1.13.3  
six 1.12.0  


## Download Checkpoints and Test
We provide a checkpoint of adversarial Puzzle Mix with PreActResNet18 trained on CIFAR-100. The model has 80.34% clean test accuracy and 42.89% accuracy against FGSM with 8/255 l-infinity epsilon-ball.

CIFAR-100 dataset will automatically be downloaded at ```[data_path]```. To test corruption robusetness, download the dataset at [here](https://github.com/hendrycks/robustness). Note that the corruption dataset should be downloaded at ```[data_path]``` with the folder name of Cifar100-C (for CIFAR100) and tiny-imagenet-200-C (for Tiny-ImageNet).

To test the model, run:
```
cd checkpoint   
python test_robust.py --ckpt preactresnet18 --datapath [data_path]
```

The other models trained with Puzzle Mix can be also downloaded:

Dataset | Model  | Method | Description | Model file
-- | -- | -- | --  | -- 
CIFAR-100 | WRN-28-10 | Puzzle Mix \[Table 2\] | 84.0% (top-1) | [drive](https://drive.google.com/drive/folders/1_vOVNYcoXFNCpTzSjmP6iKwM4I8kebNM?usp=sharing)
CIFAR-100 | WRN-28-10 | Puzzle Mix + Adv training [Table 2] | 84.0% (Top-1) / 52.8% (FGSM) | [drive](https://drive.google.com/drive/folders/1rk7pv4ov6zXjP83SmsdYFcm8J_FDJxI-?usp=sharing)
CIFAR-100 | WRN-28-10 | Puzzle Mix + Augmentation [Table 7] | 83.7% (Top-1) / 71.1% (CIFAR100-C) | [drive](https://drive.google.com/drive/folders/1G0ACJzfRGLS7-1jTbflLuMCW7QGDYrxs?usp=sharing)
CIFAR-100 | PreActResNet-18 | Puzzle Mix \[Table 3\] | 80.4% (Top-1)  | [drive](https://drive.google.com/drive/folders/1qBLhcUsicVFZi5sxEWTXI0O07QObG5dH?usp=sharing)
CIFAR-100 | PreActResNet-18 | Puzzle Mix + Adv training [Table 3] | 80.2% (Top-1) / 42.9% (FGSM)  | [drive](https://drive.google.com/drive/folders/159nUxYN58OYXtRnQSynt8iEJ0BmlXaj9?usp=sharing)
Tiny-ImageNet | PreActResNet-18 | Puzzle Mix [Table 4] | 63.9% (Top-1) | [drive](https://drive.google.com/drive/folders/1jxCib7eSoKBthNGyke7lQuVHgkl21mtZ?usp=sharing)

Also, we provide a jupyter notebook, **Visualization.ipynb**, by which users can visualize Puzzle Mix results with image samples.   

## Reproducing the results
Detailed descriptions of arguments are provided in ```main.py```. Below are some of the examples for reproducing the experimental results. 


### ImageNet
To test with ImageNet, please refer to [```./imagenet_fast```](https://github.com/snu-mllab/PuzzleMix/tree/master/imagenet_fast) or [```./imagenet```](https://github.com/snu-mllab/PuzzleMix/tree/master/imagenet) (for 300 epochs training). ```./imagenet``` contains the most concise version of Puzzle Mix training code.

### CIFAR-100
Dataset will be downloaded at ```[data_path]``` and the results will be saved at ```[save_path]```. If you want to run codes without saving results, please set ```--log_off True```.

- To reproduce **Puzzle Mix with PreActResNet18 for 1200 epochs**, run:
```
python main.py --dataset cifar100 --data_dir [data_path] --root_dir [save_path] --labels_per_class 500 --arch preactresnet18  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 1200 --schedule 400 800 --gammas 0.1 0.1 --train mixup --mixup_alpha 1.0 --graph True --n_labels 3 --eta 0.2 --beta 1.2 --gamma 0.5 --neigh_size 4 --transport True --t_size 4 --t_eps 0.8
```

- To reproduce **Puzzle Mix with PreActResNet18 for 600 epochs**, run:
```
python main.py --dataset cifar100 --data_dir [data_path] --root_dir [save_path] --labels_per_class 500 --arch preactresnet18  --learning_rate 0.2 --momentum 0.9 --decay 0.0001 --epochs 600 --schedule 350 500 --gammas 0.1 0.1 --train mixup --mixup_alpha 1.0 --graph True --n_labels 3 --eta 0.2 --beta 1.2 --gamma 0.5 --neigh_size 4 --transport True --t_size 4 --t_eps 0.8
```

- To reproduce **adversarial Puzzle Mix with PreActResNet18 for 1200 epochs**, run:
```
python main.py --dataset cifar100 --data_dir [data_path] --root_dir [save_path] --labels_per_class 500 --arch preactresnet18  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 1200 --schedule 400 800 --gammas 0.1 0.1 --train mixup --mixup_alpha 1.0 --graph True --n_labels 3 --eta 0.2 --beta 1.2 --gamma 0.5 --neigh_size 4 --transport True --t_size 4 --t_eps 0.8 --adv_p 0.1 --adv_eps 10.0
```

Below are commands to reproduce baselines.

- To reproduce **Vanilla with PreActResNet18 for 1200 epochs**, run:
```
python main.py --dataset cifar100 --data_dir [data_path] --root_dir [save_path] --labels_per_class 500 --arch preactresnet18  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 1200 --schedule 400 800 --gammas 0.1 0.1 --train vanilla
```

- To reproduce **input mixup with PreActResNet18 for 1200 epochs**, run:
```
python main.py --dataset cifar100 --data_dir [data_path] --root_dir [save_path] --labels_per_class 500 --arch preactresnet18  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 1200 --schedule 400 800 --gammas 0.1 0.1 --train mixup --mixup_alpha 1.0
```

- To reproduce **manifold mixup with PreActResNet18 for 1200 epochs**, run:
```
python main.py --dataset cifar100 --data_dir [data_path] --root_dir [save_path] --labels_per_class 500 --arch preactresnet18  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 1200 --schedule 400 800 --gammas 0.1 0.1 --train mixup_hidden --mixup_alpha 2.0
```

- To reproduce **CutMix with PreActResNet18 for 1200 epochs**, run:
```
python main.py --dataset cifar100 --data_dir [data_path] --root_dir [save_path] --labels_per_class 500 --arch preactresnet18  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 1200 --schedule 400 800 --gammas 0.1 0.1 --train mixup --mixup_alpha 1.0 --box True
```


For **WRN28_10 with 400 epoch**, set ```--arch wrn28_10```, ```--epochs 400```, and ```--schedule 200 300```. For **WRN28_10 with 200 epoch**, set ```--epochs 200```, ```--schedule 120 170```, and ```--learning_rate 0.2```.


### Tiny-Imagenet-200

#### Download dataset
The following process is forked from ([link](https://github.com/vikasverma1077/manifold_mixup/tree/master/supervised)).

1. Download the zipped data from https://tiny-imagenet.herokuapp.com/  
2. If not already exiting, create a subfolder "data" in root folder "PuzzleMix"  
3. Extract the zipped data in folder PuzzleMix/data  
4. Run the following script (This will arange the validation data in the format required by the pytorch loader)
```
python load_data.py
```

- To reproduce **Puzzle Mix with PreActResNet18 for 1200 epochs**, run:
```
python main.py --dataset tiny-imagenet-200 --data_dir [data_path] --root_dir [save_path] --labels_per_class 500 --arch preactresnet18  --learning_rate 0.2 --momentum 0.9 --decay 0.0001 --epochs 1200 --schedule 600 900 --gammas 0.1 0.1 --train mixup --mixup_alpha 1.0 --graph True --n_labels 3 --eta 0.2 --beta 1.2 --gamma 0.5 --neigh_size 4 --transport True --t_eps 0.8 --clean_lam 1
```

- To reproduce **Puzzle Mix with PreActResNet18 for 600 epochs**, run:
```
python main.py --dataset tiny-imagenet-200 --data_dir [data_path] --root_dir [save_path] --labels_per_class 500 --arch preactresnet18  --learning_rate 0.2 --momentum 0.9 --decay 0.0001 --epochs 600 --schedule 300 450 --gammas 0.1 0.1 --train mixup --mixup_alpha 1.0 --graph True --n_labels 3 --eta 0.2 --beta 1.2 --gamma 0.5 --neigh_size 4 --transport True --t_eps 0.8 --clean_lam 1
```

- To reproduce **adversarial Puzzle Mix with PreActResNet18 for 1200 epochs**, run:
```
python main.py --dataset tiny-imagenet-200 --data_dir [data_path] --root_dir [save_path] --labels_per_class 500 --arch preactresnet18  --learning_rate 0.2 --momentum 0.9 --decay 0.0001 --epochs 1200 --schedule 600 900 --gammas 0.1 0.1 --train mixup --mixup_alpha 1.0 --graph True --n_labels 3 --eta 0.2 --beta 1.2 --gamma 0.5 --neigh_size 4 --transport True --t_eps 0.8 --adv_p 0.15 --adv_eps 10.0 --clean_lam 1
```

- To reproduce **adversarial Puzzle Mix with PreActResNet18 for 600 epochs**, run:
```
python main.py --dataset tiny-imagenet-200 --data_dir [data_path] --root_dir [save_path] --labels_per_class 500 --arch preactresnet18  --learning_rate 0.2 --momentum 0.9 --decay 0.0001 --epochs 600 --schedule 300 450 --gammas 0.1 0.1 --train mixup --mixup_alpha 1.0 --graph True --n_labels 3 --eta 0.2 --beta 1.2 --gamma 0.5 --neigh_size 4 --transport True --t_eps 0.8 --adv_p 0.15 --adv_eps 10.0 --clean_lam 1
```


Below are commands to reproduce baselines.

- To reproduce **Vanilla with PreActResNet18 for 1200 epochs**, run:
```
python main.py --dataset tiny-imagenet-200 --data_dir [data_path] --root_dir [save_path] --labels_per_class 500 --arch preactresnet18  --learning_rate 0.2 --momentum 0.9 --decay 0.0001 --epochs 1200 --schedule 600 900 --gammas 0.1 0.1 --train vanilla
```

- To reproduce **input mixup with PreActResNet18 for 1200 epochs**, run:
```
python main.py --dataset tiny-imagenet-200 --data_dir [data_path] --root_dir [save_path] --labels_per_class 500 --arch preactresnet18  --learning_rate 0.2 --momentum 0.9 --decay 0.0001 --epochs 1200 --schedule 600 900 --gammas 0.1 0.1 --train mixup --mixup_alpha 0.2
```

- To reproduce **manifold mixup with PreActResNet18 for 1200 epochs**, run:
```
python main.py --dataset tiny-imagenet-200 --data_dir [data_path] --root_dir [save_path] --labels_per_class 500 --arch preactresnet18  --learning_rate 0.2 --momentum 0.9 --decay 0.0001 --epochs 1200 --schedule 600 900 --gammas 0.1 0.1 --train mixup_hidden --mixup_alpha 0.2
```

- To reproduce **CutMix with PreActResNet18 for 1200 epochs**, run:
```
python main.py --dataset tiny-imagenet-200 --data_dir [data_path] --root_dir [save_path] --labels_per_class 500 --arch preactresnet18  --learning_rate 0.2 --momentum 0.9 --decay 0.0001 --epochs 1200 --schedule 600 900 --gammas 0.1 0.1 --train mixup --mixup_alpha 0.2 --box True
```


## License
MIT License
