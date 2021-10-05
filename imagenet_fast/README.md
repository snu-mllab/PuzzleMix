# Puzzle Mix: Exploiting Saliency and Local Statistics for Optimal Mixup (ImageNet)
This is the code for fast ImageNet training based on [Apex](https://github.com/NVIDIA/apex). The training and data preprocessing codes are borrowed from `Fast is better than free: revisiting adversarial training' ([link](https://github.com/anonymous-sushi-armadillo/fast_is_better_than_free_imagenet)). For 300 epochs training, please refer to ```../cutmix```

## Requirements
1. Install the required python packages. All packages can be installed by running the following command:
```bash
pip install -r requirements.txt
```
2. Install [Apex](https://github.com/NVIDIA/apex) to use half precision speedup. 


## Preparing ImageNet Data
1. Download and prepare the ImageNet dataset. You can use [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh), 
provided by the PyTorch repository, to move the validation subset to the labeled subfolders.
2. Prepare resized versions of the ImageNet dataset, you can use `resize.py` provided in this repository. 

## Reproducing the results
To reproduce the results from the paper, modify ```DATA160``` and ```DATA352``` (in run_fast.sh) with your own ```data path``` from `resize.py`.
Then run 
```
run_fast.sh
``` 
This script runs the main code `main_fast.py` using the configurations provided in the `configs/` folder. All parameters can be modified by adjusting the configuration files in the `configs/` folder. To evaluate the trained model, run `run_test.sh`.
