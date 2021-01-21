# DSDN
Dangerous Sound Detection Network module for gunshot detection using VGG16 and transformation to spectrograms

![](https://i.imgur.com/7L6713J.png)

## Folder structure:
- `audio_set` - samples of audio
- `spec_set` - samples of generated spectograms
- `data` - csv files from [AudioSet](https://research.google.com/audioset/download.html) site
- `models` - trained models and plots

## Requirements
- `numpy==1.19.5`
- `opencv-python==4.5.1.48`
- `tensorflow==2.4.0`
- `pytube==10.0.0`
- `skimage==10.0.0`
- `keras==2.4.3`
- `moviepy==1.0.3`
- `librosa==0.8.0`

## Required Data
To evaluate/train model, you will need to download the required datasets:
* [Gunshot](https://research.google.com/audioset/dataset/gunshot_gunfire.html)
* [Fireworks](https://research.google.com/audioset/dataset/fireworks.html)
* [Clapping](https://research.google.com/audioset/dataset/clapping.html)
* [Screaming](https://research.google.com/audioset/dataset/screaming.html)

Or, you can easily extract all required data (transformed to spectrograms) using "extract_audio.py" module as below:
```
python extract_audio.py --labels_path data/labels.cvs --data_path data/data.csv --label [gunfire | screamming | clapping | ...]
 ```

## Training
Training using ours settings:
```
python train.py --spectrograms_dir path//to//dir//
```
Example of training with different settings:
```
python train.py --s path//to//dir// -- resize 64 --epochs 100 --batch_size 32 --class_number 5
```

## Prediction
To predict trained model you will have to load it with Keras:
```
model = load_model(MODEL_PATH)
```
and after data tranformation, evalute using predict method from Keras:
```
predict = model.predict(data)
```