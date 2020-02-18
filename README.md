# Puzzle Mix: Exploiting Saliency and Local Statistics for Optimal Mixup

This is the code for the paper "Puzzle Mix: Exploiting Saliency and Local Statistics for Optimal Mixup" submitted to ICML'20. Some parts of the codes are borrowed from manifold mixup (https://github.com/vikasverma1077/manifold_mixup/tree/master/supervised).

## Requirements
This code has been tested with  
python 3.6.8  
pytorch 1.1.0  
torchvision 0.3.0
gco-wrapper (https://github.com/Borda/pyGCO)

matplotlib==3.0.2  
numpy==1.15.4  
pandas==0.23.4  
Pillow==5.4.1  
scipy==1.1.0  
seaborn==0.9.0  
six==1.12.0  

## Reproducing the results
Detailed descriptions of arguments are provided in ```main.py```. Below are some of the examples for reproducing the experimental results. 

### CIFAR-100
Dataset will be downloaded at ```[data_path]``` and results will be saved at ```[save_path]```.

To reproduce **Puzzle Mix with PreActResNet18 for 1200 epochs**, run:
```
python main.py --dataset cifar100 --data_dir [data_path] --root_dir [save_path] --labels_per_class 500 --arch preactresnet18  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 1200 --schedule 400 800 --gammas 0.1 0.1 --train mixup --mixup_alpha 1.0 --graph True --n_labels 3 --eta 0.2 --beta 1.2 --gamma 0.5 --neigh_size 4 --transport True --t_size 4 --t_eps 0.8
```

To reproduce **Puzzle Mix with PreActResNet18 for 600 epochs**, run:
```
python main.py --dataset cifar100 --data_dir [data_path] --root_dir [save_path] --labels_per_class 500 --arch preactresnet18  --learning_rate 0.2 --momentum 0.9 --decay 0.0001 --epochs 600 --schedule 350 500 --gammas 0.1 0.1 --train mixup --mixup_alpha 1.0 --graph True --n_labels 3 --eta 0.2 --beta 1.2 --gamma 0.5 --neigh_size 4 --transport True --t_size 4 --t_eps 0.8
```

To reproduce **adversarial Puzzle Mix with PreActResNet18 for 1200 epochs**, run:
```
python main.py --dataset cifar100 --data_dir [data_path] --root_dir [save_path] --labels_per_class 500 --arch preactresnet18  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 1200 --schedule 400 800 --gammas 0.1 0.1 --train mixup --mixup_alpha 1.0 --graph True --n_labels 3 --eta 0.2 --beta 1.2 --gamma 0.5 --neigh_size 4 --transport True --t_size 4 --t_eps 0.8 --adv_p 0.1 --adv_eps 10.0
```

Below are commands to reproduce baselines.

To reproduce **Vanilla with PreActResNet18 for 1200 epochs**, run:
```
python main.py --dataset cifar100 --data_dir [data_path] --root_dir [save_path] --labels_per_class 500 --arch preactresnet18  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 1200 --schedule 400 800 --gammas 0.1 0.1 --train vanilla
```

To reproduce **input mixup with PreActResNet18 for 1200 epochs**, run:
```
python main.py --dataset cifar100 --data_dir [data_path] --root_dir [save_path] --labels_per_class 500 --arch preactresnet18  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 1200 --schedule 400 800 --gammas 0.1 0.1 --train mixup --mixup_alpha 1.0
```

To reproduce **manifold mixup with PreActResNet18 for 1200 epochs**, run:
```
python main.py --dataset cifar100 --data_dir [data_path] --root_dir [save_path] --labels_per_class 500 --arch preactresnet18  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 1200 --schedule 400 800 --gammas 0.1 0.1 --train mixup_hidden --mixup_alpha 2.0
```

To reproduce **CutMix with PreActResNet18 for 1200 epochs**, run:
```
python main.py --dataset cifar100 --data_dir [data_path] --root_dir [save_path] --labels_per_class 500 --arch preactresnet18  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 1200 --schedule 400 800 --gammas 0.1 0.1 --train mixup --mixup_alpha 1.0 --box True
```

For WRN28_10, we need to change ```--arch wrn28_10```, ```--epochs 400```, and ```--schedule 200 300```. For 200 epoch, we set ```--epochs 200```, ```--schedule 120 170```, and ```--learning_rate 0.2'''.




### How to run experiments for Tiny-Imagenet-200

1.Download the zipped data from https://tiny-imagenet.herokuapp.com/  
2.If not already exiting, create a subfolder "data" in root folder "manifold_mixup"  
3.Extract the zipped data in folder manifold_mixup/data  
4.Run the following script (This will arange the validation data in the format required by the pytorch loader)
```
python utils.py
```

5. Run the following commands  
#### No mixup Preactresnet18
```
python main.py --dataset tiny-imagenet-200 --data_dir data/tiny-imagenet-200/ --root_dir experiments/ --labels_per_class 500 --arch preactresnet18  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 1000 1500 --gammas 0.1 0.1 --train vanilla 
```

####  Mixup Preactresnet18
```
python main.py --dataset tiny-imagenet-200 --data_dir data/tiny-imagenet-200/ --root_dir experiments/ --labels_per_class 500 --arch preactresnet18  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 1000 1500 --gammas 0.1 0.1 --train mixup --mixup_alpha 0.2
```

#### Manifold mixup Preactresnet18
```
python main.py --dataset tiny-imagenet-200 --data_dir data/tiny-imagenet-200/ --root_dir experiments/ --labels_per_class 500 --arch preactresnet18  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 1000 1500 --gammas 0.1 0.1 --train mixup_hidden --mixup_alpha 0.2

```





