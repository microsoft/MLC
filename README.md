# Meta Label Correction for Noisy Label Learning

This repository contains the source code for the AAAI paper "Meta Label Correction for Noisy Label Learning".

## Example

For experiments on CIFAR, run MLC with UNIF noise and a noise level of 0.4 by executing
```python
python3 main.py --dataset cifar10 --optimizer sgd --bs 100 --corruption_type unif --corruption_level 0.4 --gold_fraction 0.02 --epochs 120 --main_lr 0.1 --meta_lr 3e-4 --runid cifar10_run  --cls_dim 128
```

Refer to ```python3 main.py --help``` for a detailed explanations of all applicable arguments.

## Citation

If you are find MLC useful, please cite the following paper

```
@article{zheng2021mlc,
  title={Meta Label Correction for Noisy Label Learning},
  author={Zheng, Guoqing and Awadallah, Ahmed Hassan and Dumais, Susan},  
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  year={2021},
}
```

This repository is released under MIT License. (See [LICENSE](LICENSE))

