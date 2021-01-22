# DIDN

Dangerous Item Detection Network based trained with [YOLOv3](https://pjreddie.com/darknet/yolo/) and translated to Tensorflow library with usage of [tool used for translation](https://github.com/hunglc007/tensorflow-yolov4-tflite)

## Folder structure:

- `model` - space to place trained model
- `test_set` - sample video for testing

## Installing dependencies

### Pip

```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```

## Running

Detection of weapon from sample video

```python
python detect.py
```
