import cv2
import numpy as np


def convert_to_optical_flow(prev, sec, mask):
    """
    Converts two images to optical flow
    Args:
        prev: first frame
        sec: second frame
        mask: mask

    Returns:
        optical flow in rgb
    """
    flow = cv2.calcOpticalFlowFarneback(prev, sec, None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    return rgb
