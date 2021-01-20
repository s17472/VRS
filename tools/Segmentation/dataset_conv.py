import argparse
import os
from imutils import paths
from keras_segmentation import convert_to_segmentation


def create_dir(dir):
    """
    Creates directory if not exist
    Args:
        dir: path of the directory
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def convert(args):
    """
    Converts dataset of images to dataset of segmentation images
    Args:
        args: args
    """
    # dirs violence and notViolence
    type_dirs = os.listdir(args.path)
    for type_dir in type_dirs:
        # dirs with images
        images_dirs = os.listdir(args.path + type_dir + '/')
        for images_dir in images_dirs:
            current_path = args.path + "{}/{}/".format(type_dir, images_dir)
            print("Current dir:", current_path)

            # gets images
            images = list(paths.list_files(current_path))

            for img_n, image in enumerate(images):
                save_path = args.path_save + "{}/{}/".format(type_dir, images_dir)
                image_path = current_path + image
                create_dir(save_path)

                convert_to_segmentation(image_path, save_path)
                print("{}/{}".format(img_n, len(images) - 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help="dataset to convert")
    parser.add_argument('--path_save', help="converted dataset dir")
    args = parser.parse_args()

    convert(args)
