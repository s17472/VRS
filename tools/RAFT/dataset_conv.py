import sys
from raft_optical_flow import load_image, flo_to_image, get_optical_flow

sys.path.append('core')

import argparse
import os
import numpy as np
import torch
from PIL import Image
from imutils import paths
from raft import RAFT


DEVICE = 'cuda'


def save_images(images, path):
    """
    Save images in path
    Args:
        images: images to save
        path: path of directory to save
    """
    if not os.path.exists(path):
        os.makedirs(path)

    for i, image in enumerate(images):
        im = Image.fromarray(np.uint8(image))
        im.save(path + "{}.jpg".format(i))


def convert(args):
    """
    Converts dataset of images to dataset of optical flow images
    Args:
        args: args
    """
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        # dirs violence and notViolence
        type_dirs = os.listdir(args.path)
        for type_dir in type_dirs:
            # dirs with images
            images_dirs = os.listdir(args.path + type_dir + '/')
            for images_dir in images_dirs:
                current_path = args.path + "{}/{}".format(type_dir, images_dir)
                print("Current dir:", current_path)

                # gets images
                images = list(paths.list_files(current_path))

                flows = []
                for imfile1, imfile2 in zip(images[:-1], images[1:]):
                    flow_up = get_optical_flow(imfile1, imfile2)
                    flow = flo_to_image(flow_up)
                    flows.append(flow)
                    print("{}/{}".format(len(flows), len(images)-1))

                save_images(flows, args.path_save + "{}/{}/".format(type_dir, images_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset to convert")
    parser.add_argument('--path_save', help="converted dataset dir")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    convert(args)
