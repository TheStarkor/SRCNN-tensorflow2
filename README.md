# SRCNN-tensorflow
The original paper is [Learning a Deep Convolutional Network for Image Super-Resolution](https://arxiv.org/abs/1501.00092)  
Dataset은 General-100 을 사용했고[2] 학습 과정은 다음 colab에 있습니다. 
- [prepare_data](https://colab.research.google.com/drive/1hEyPcukzc_K5w2WLS5BFhkOIcMmFbxQa#scrollTo=ErzuyS4tU-3D)
- [main](https://colab.research.google.com/drive/17yuR0DYtRO3S4Ws2OZS-mPMhtH0lQgOS#scrollTo=6Qa3LgnT7X9N)

## Prerequisites
- Anaconda
- Dataset

## Usage
```
$ conda env create -f requirements.yaml
$ python prepare_data.py
$ main.py
```

## Results
![results](https://user-images.githubusercontent.com/45455072/82464619-4138d800-9af9-11ea-88c4-dc9e40d0c6de.png)

## References
[1] [MarkPrecursor/SRCNN-keras](https://github.com/MarkPrecursor/SRCNN-keras)
    - I have followed and learned training process and structure of this repository.
[2] [General-100](https://drive.google.com/file/d/0B7tU5Pj1dfCMVVdJelZqV0prWnM/view)