import sys

import cv2

sys.path.append('core')
import argparse
import numpy as np
import torch

from utils import flow_viz
from utils.utils import InputPadder
from raft import RAFT


args = argparse.Namespace()
args.alternate_corr = False
args.mixed_precision = False
args.small = False
args.model = 'raft-sintel.pth'

model = torch.nn.DataParallel(RAFT(args))
model.load_state_dict(torch.load(args.model))

model = model.module
model.to('cuda')
model.eval()


def load_image(img):
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to('cuda')


def flo_to_image(flo):
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    flo = flow_viz.flow_to_image(flo)
    return flo


def reshape(frame, size):
    frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    frame = np.reshape(frame, (size, size, 3))
    return frame


def convert_to_optical_flow(prev, sec, size):
    image1 = reshape(prev, size)
    image2 = reshape(sec, size)

    image1 = load_image(image1)
    image2 = load_image(image2)

    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)

    with torch.no_grad():
        _, flow_up = model(image1, image2, iters=32, test_mode=True)

    flow = flo_to_image(flow_up)

    return flow
