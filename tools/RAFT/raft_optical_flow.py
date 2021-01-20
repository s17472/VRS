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
args.model = 'raft-model.pth'

model = torch.nn.DataParallel(RAFT(args))
model.load_state_dict(torch.load(args.model))

model = model.module
model.to('cuda')
model.eval()


def load_image(img):
    """
    Load image as np array
    Args:
        img: path to the image

    Returns:
        image in np array
    """
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to('cuda')


def flo_to_image(flo):
    """
    Converts flow to rgb
    Args:
        flo: flow object

    Returns:
        image of the flow
    """
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    flo = flow_viz.flow_to_image(flo)
    return flo


def reshape(frame, size):
    """
    Reshape frame
    Args:
        frame: frame to reshape
        size: size of the reshape

    Returns:
        reshaped frames
    """
    frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    frame = np.reshape(frame, (size, size, 3))
    return frame


def get_optical_flow(img1, img2):
    """
    Gets optical flow
    Args:
        img1: first image
        img2: second image

    Returns:
        optical flow
    """
    image1 = load_image(img1)
    image2 = load_image(img2)

    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)

    with torch.no_grad():
        _, flow_up = model(image1, image2, iters=32, test_mode=True)
    return flow_up


def convert_to_optical_flow(prev, sec, size):
    """
    Converts two images to optical flow
    Args:
        prev: first frame
        sec: second frame
        size: size of the resize

    Returns:
        optical flow image
    """
    prev = reshape(prev, size)
    sec = reshape(sec, size)

    flow_up = get_optical_flow(prev, sec)
    flow = flo_to_image(flow_up)

    return flow
