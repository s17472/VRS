# VRN
Violence Recognition Network that uses VGG16 network as base and LSTM as one of the top layers.

![](https://i.imgur.com/L2GsPSA.png)

## Requirements
- `tensorflow==2.4.0`
- `opencv-python==4.2.0.34`
- `h5py==2.10.0`

## Required Data
To evaluate/train model, you will need to download the required dataset:
* [RWF-2000](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection/blob/master/Agreement%20Sheet.pdf)

If you want to evaluate/train model on different dataset, you will have to create directory structure as below:

```bash
├── dataset
│   ├── Fight
│   │   ├── video0.avi
│   │   └── video1.avi
│   │   ... 
│   └── NonFight
│       ├── video0.avi
│       └── video1.avi
│        ... 
```

## Training
Training using our settings:
```
python vrn.py
```
All the settings like directory paths or hyperparameters you have to set inside vrn.py file
