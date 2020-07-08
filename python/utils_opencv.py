# code borrowed from https://github.com/svip-lab/impersonator/blob/master/utils/cv_utils.py

import cv2
from matplotlib import pyplot as plt
import numpy as np


HMR_IMG_SIZE = 224
IMG_SIZE = 256


def read_cv2_img(path):
    """
    Read color images
    :param path: Path to image
    :return: Only returns color images
    """
    img = cv2.imread(path, -1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def save_cv2_img(img, path, image_size=None, normalize=False):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # print('normalize = {}'.format(normalize))

    if image_size is not None:
        img = cv2.resize(img, (image_size, image_size))

    if normalize:
        img = (img + 1) / 2.0 * 255
        img = img.astype(np.uint8)

    cv2.imwrite(path, img)
    return img


def transform_img(image, image_size, transpose=False):
    image = cv2.resize(image, (image_size, image_size))
    image = image.astype(np.float32)
    image /= 255.0

    if transpose:
        image = image.transpose((2, 0, 1))

    return image


def resize_img_with_scale(img, scale_factor):
    new_size = (np.floor(np.array(img.shape[0:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])
    ]
    return new_img, actual_factor

