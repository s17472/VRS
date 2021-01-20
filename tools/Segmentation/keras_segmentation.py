from keras_segmentation.pretrained import pspnet_101_voc12


def convert_to_segmentation(image_path, save_path):
    """
    Convert image with segmentation model
    Args:
        image_path: path to image to convert
        save_path: save path
    """
    # load the pretrained model trained on Pascal VOC 2012 dataset
    model = pspnet_101_voc12()
    print(model)

    model.predict_segmentation(inp=image_path, out_fname=save_path)
