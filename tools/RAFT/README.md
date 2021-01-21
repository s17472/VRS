# Optical flow with RAFT
Dataset tranformation using Optical Flow [RAFT](https://github.com/princeton-vl/RAFT) method

![](https://i.imgur.com/IMlqSU9.png)
## Requirements
You can find all requirements in `requirements-raft.txt` file

## Required Data
To convert your dataset to Optical Flow, you will need to create directory strucutre as below:

```bash
├── dataset
│   ├── Fight
│   │   ├── 0
│   │   │   ├── frame0.png
│   │   │   ├── frame1.png
│   │   │   ... 
│   │   └── 1
│   │   │   ├── frame0.png
│   │   │   ├── frame1.png
│   │       ... 
│   └── NonFight
│   │   ├── 0
│   │   │   ├── frame0.png
│   │   │   ├── frame1.png
│   │   │   ... 
│   │   └── 1
│   │   │   ├── frame0.png
│   │   │   ├── frame1.png
│   │       ... 
```

## Using
```
python dataset_conv.py --model raft-model.pth --path path//to//dataset// --path_save path//to//save//dir//
```