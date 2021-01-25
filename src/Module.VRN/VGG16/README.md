# VGG16
VGG16 network used as base for VRN.  
Created from ground up with an additional option to fine tune with imagenet weights. 


## Requirements
- `tensorflow==2.4.0`

## Required Data
To evaluate/train model, you will need to download the required dataset:
* [RWF-2000](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection/blob/master/Agreement%20Sheet.pdf)

Have in mind you have to extract frames from that videos to image format.

If you want to evaluate/train model on different dataset, you will have to create directory structure as below:

```bash
├── dataset
│   ├── train
│   │   ├── Fight
│   │   │   ├── frame0.jpg
│   │   │   ├── frame1.jpg
│   │   │   ... 
│   │   └── NonFight
│   │       ├── frame0.jpg
│   │       ├── frame1.jpg
│   │       ... 
│   └── val
│       ├── Fight
│       │   ├── frame0.jpg
│       │   ├── frame1.jpg
│       │   ... 
│       └── NonFight
│           ├── frame0.jpg
│           ├── frame1.jpg
│           ... 
```

## Training
Training using our settings:
```
python vgg16.py
```
All the settings like directory paths or hyperparameters you have to set inside vgg16.py file

