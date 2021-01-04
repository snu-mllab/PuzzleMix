## ImageNet for 300 epochs

In this repository, we test Puzzle Mix on the experimental setting of [CutMix](https://github.com/clovaai/CutMix-PyTorch). 
Specifically, we train ResNet-50 for 300 epochs with Puzzle Mix and the result is as follows. The code is tested on PyTorch 1.7.

 Method | Top-1 Error | Model file
 -- | -- | --
 ResNet-50 [[CVPR'16](https://arxiv.org/abs/1512.03385)] (baseline) | 23.68 | [model](https://www.dropbox.com/sh/phwbbrtadrclpnx/AAA9QUW9G_xvBdI-mDiIzP_Ha?dl=0)
 ResNet-50 + Mixup [[ICLR'18](https://arxiv.org/abs/1710.09412)] | 22.58 | [model](https://www.dropbox.com/sh/g64c8bda61n12if/AACyaTZnku_Sgibc9UvOSblNa?dl=0)
 ResNet-50 + Manifold Mixup [[ICML'19](https://arxiv.org/abs/1806.05236)] | 22.50 | [model](https://www.dropbox.com/sh/bjardjje11pti0g/AABFGW0gNrNE8o8TqUf4-SYSa?dl=0)
 ResNet-50 + Cutout [[arXiv'17](https://arxiv.org/abs/1708.04552)] | 22.93 | [model](https://www.dropbox.com/sh/ln8zk2z7zt2h1en/AAA7z8xTBlzz7Ofbd5L7oTnTa?dl=0)
 ResNet-50 + AutoAugment [[CVPR'19](https://arxiv.org/abs/1805.09501)] | 22.40* | -
 ResNet-50 + DropBlock [[NeurIPS'18](https://arxiv.org/abs/1810.12890)] | 21.87* | -
 ResNet-50 + CutMix | 21.40 | [model](https://www.dropbox.com/sh/w8dvfgdc3eirivf/AABnGcTO9wao9xVGWwqsXRala?dl=0)
 ResNet-50 + Feature CutMix | 21.80 | [model](https://www.dropbox.com/sh/zj1wptsg0hwqf0k/AABRNzvjFmIS7_vOEQkqb6T4a?dl=0)
 ResNet-50 + **Puzzle Mix** | **21.24** | -

To download the trained model, please run the following.
```
pip install gdown
gdown https://drive.google.com/uc?id=1w7mnpJO_mdMpBvrqGuetuDRnrT8xUbb1 -O imagenet_puzzlemix.pth.tar
```

To test the model, please run the following.
```
python test.py --pretrained imagenet_puzzlemix.pth.tar
```

To train the model, run the following (we used 4 GPUs). 
```
python train.py \
--net_type resnet \
--dataset imagenet \
--batch_size 256 \
--lr 0.1 \
--depth 50 \
--epochs 300 \
--expname ResNet50 \
-j 40 \
--beta 1.0 \
--mixup_prob 1.0 \
--method puzzle \
--transport True \
--no-verbose
```
