import argparse
import os
import numpy as np
from PIL import Image
from imutils import paths
from opencv_optical_flow import convert_to_optical_flow


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


def load_image(img_file):
    """
     Load image as np array
     Args:
         img_file: path to the image

     Returns:
         image in np array
     """
    return np.array(Image.open(img_file)).astype(np.uint8)


def convert(args):
    """
    Converts dataset of images to dataset of optical flow images
    Args:
        args: args
    """
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

            first_image = load_image(images[0])
            mask = np.zeros_like(first_image)
            mask[..., 1] = 255

            flows = []
            for imfile1, imfile2 in zip(images[:-1], images[1:]):
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)

                flow = convert_to_optical_flow(image1, image2, mask)
                flows.append(flow)
                print("{}/{}".format(len(flows), len(images) - 1))
            save_images(flows, args.path_save + "{}/{}/".format(type_dir, images_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help="dataset to convert")
    parser.add_argument('--path_save', help="converted dataset dir")
    args = parser.parse_args()

    convert(args)
