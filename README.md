# SRCNN-tensorflow
The original paper is [Learning a Deep Convolutional Network for Image Super-Resolution](https://arxiv.org/abs/1501.00092). Dataset is General-100[2]
![SRCNN](https://user-images.githubusercontent.com/45455072/82465244-08e5c980-9afa-11ea-8db2-0458af007012.png)  

## Prerequisites
- Anaconda
- Dataset

## Usage
```
$ conda env create -f requirements.yaml
$ python prepare_data.py
$ main.py
```

## Colab
- [prepare_data](https://colab.research.google.com/drive/1hEyPcukzc_K5w2WLS5BFhkOIcMmFbxQa#scrollTo=ErzuyS4tU-3D)
- [main](https://colab.research.google.com/drive/17yuR0DYtRO3S4Ws2OZS-mPMhtH0lQgOS#scrollTo=6Qa3LgnT7X9N)


## Results
![set5](https://user-images.githubusercontent.com/45455072/82467489-ae01a180-9afc-11ea-9513-3b7ba455346f.png)  
![set14](https://user-images.githubusercontent.com/45455072/82467631-d5586e80-9afc-11ea-8d90-8d274f4cd1e7.png)


## References
[1] [MarkPrecursor/SRCNN-keras](https://github.com/MarkPrecursor/SRCNN-keras)
    - I have followed and learned training process and structure of this repository.  
[2] [General-100](https://drive.google.com/file/d/0B7tU5Pj1dfCMVVdJelZqV0prWnM/view)
