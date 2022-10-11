# Frequency-Domain Deep Guided Image Denoising (TMM 2022)
Zehua Sheng, Xiongwei Liu, Si-Yuan Cao, Hui-Liang Shen, and Huaqi Zhang

## Description
This is the code implementation of our paper "Frequency-domain deep guided image denoising" to be published in 2022 IEEE Transactions on Multimedia. This model is built in PyTorch 1.4.0 and tested on Ubuntu 16.04 (Python 3.6.13, CUDA 10.0). At present, we provide our pre-trained models on the RGB-NIR Scene Dataset. Training codes and testing codes on the rest datasets will be made available soon.


## Evaluation
We provide one demo to evaluate the example paired images from the RGB-NIR Scene Dataset.
```
# Evaluation on RGB-NIR Scene Dataset
python demo_test_single_rgbnir.py
```

## Acknowledgement
Our work is mainly evaluated on the RGB-NIR Scene Dataset, the Flash and Ambient Illuminations Dataset, and the NYU v2 Dataset. We thank the authors of the open datasets for their contributions.
