# FGN
Flow Gated Network module based on [Violence Detection project](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection)

![](https://i.imgur.com/GimUqDX.png)

## Folder structure:
- `model` - exported model in tf format
- `plots` - images with traning history
- `testset` - samples videos for testing

## Requirements
- `numpy==1.19.5`
- `opencv-python==4.5.1.48`
- `tensorflow==2.4.0`

## Required Data
To evaluate/train model, you will need to download the required dataset:
* [RWF-2000](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection/blob/master/Agreement%20Sheet.pdf)

If you want to evaluate/train model on different dataset, you will have to create directory structure as below:

```bash
├── dataset
│   ├── train
│   │   ├── Fight
│   │   │   ├── video0.avi
│   │   │   ├── video1.avi
│   │   │   ... 
│   │   └── NonFight
│   │       ├── video0.avi
│   │       ├── video1.avi
│   │       ... 
│   └── val
│       ├── Fight
│       │   ├── video0.avi
│       │   ├── video1.avi
│       │   ... 
│       └── NonFight
│           ├── video0.avi
│           ├── video1.avi
│           ... 
```

## Training
Training using our settings:
```
python train.py -m model_name -d path//to//dataset//dir//
```
Example of training with different settings:
```
python train.py -m model_name -d path//to//dataset//dir// -size 224 -frame_number 32 -epoch 100 --batch_size 16 --workers_number 8
```

## Prediction
Prediction using our trained weights:
```
python predict.py -m model_final --show
```
Prediction of first batch of the video:
```
python predict_on_batch.py -m model_final --source <source of the video>
```
