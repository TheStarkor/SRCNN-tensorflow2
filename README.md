# EDSR-tensorflow2
Tensorflow 2 implementation of [Learning a Deep Convolutional Network for Image Super-Resolution](https://arxiv.org/abs/1501.00092).
![SRCNN](https://user-images.githubusercontent.com/45455072/82465244-08e5c980-9afa-11ea-8db2-0458af007012.png)  

## Usage
```
$ python main.py [-h] N_TRAIN_DATA N_TEST_DATA BATCH_SIZE EPOCHS 
```
### DIV2K example
```
$ python main.py 800 100 16 200
```
### Prerequisites
- Python 3.7
- Tensorflow 2
- Numpy

## Directory
```
EDSR-tensorflow2
├── main.py              // main program
├── model.py             // srcnn model
├── data_generator.py    // data augmentation
└── utils.py             // psnr
```

## Sample Results
- trained by DIV2K
- test image set is Set14  

  
![sample](https://user-images.githubusercontent.com/45455072/91018344-6d00b800-e62a-11ea-843a-361c6a954340.png)
