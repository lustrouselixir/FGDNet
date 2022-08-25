# Frequency-Domain Deep Guided Image Denoising
Zehua Sheng, Xiongwei Liu, Si-Yuan Cao, Hui-Liang Shen, and Huaqi Zhang

## Description
This model is built in PyTorch 1.6.0 and tested on Ubuntu 18.04 (Python 3.6.13, CUDA 10.0). At present, we provide our pre-trained models on the RGB-NIR Scene Dataset. Training codes and testing codes on the rest datasets will be made available soon.


## Evaluation
We provide 1 demo to evaluate the example paired images from the RGB-NIR Scene Dataset.
```
# Evaluation on RGB-NIR Scene Dataset
python demo_test_single_rgbnir.py
```

## Acknowledgement
Our work is mainly evaluated on the RGB-NIR Scene Datset, the Flash and Ambient Illuminations Dataset, and the NYU v2 Dataset. We thank the authors of the open datasets for their contributions.