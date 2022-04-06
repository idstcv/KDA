# KDA
PyTorch Implementation for Our SDM'22 Paper: "Improved Knowledge Distillation via Full Kernel Matrix Transfer"

## Requirements
* Python 3.8
* PyTorch 1.6

## Usage:
KDA on CIFAR-100
```
CUDA_VISIBLE_DEVICES=0 python main_kda.py --alpha 0.04 --teacher-model /path/to/teacher  /path/to/cifar100
```

## Citation
If you use the package in your research, please cite our paper:
```
@inproceedings{qian2022kda,
  author    = {Qi Qian and
               Hao Li and
               Juhua Hu},
  title     = {Improved Knowledge Distillation via Full Kernel Matrix Transfer},
  booktitle = {SIAM International Conference on Data Mining, {SDM} 2022},
  year      = {2022}
}
```
