# Several utils used to process the images

import cv2
import numpy as np
from core.config import settings

# The default image size. MUST BE the same size you trained SpellNet
side = settings['PIPOTTER_SIDE_SPELL_NET']


def pad_to_square(im, thumbnail_size=side, color=[0, 0, 0]):
    """
    Resizes an image into a square of thubmnail_size x thumbnail_size
    :param im: numpy array of w x h x channels image
    :param thumbnail_size: side of the square to resie de image into
    :param color: color for the background, black by default
    :return: a thumbnail_size x thumbnail_size resized image
    """
    squared = np.zeros([side, side, 3], np.uint8) # In case we receive an exotic image size
    old_size = im.shape[:2]  # old_size is in (height, width) format
    ratio = float(thumbnail_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    try:
        # new_size should be in (width, height) format
        im = cv2.resize(im, (new_size[1], new_size[0]))

        delta_w = thumbnail_size - new_size[1]
        delta_h = thumbnail_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        squared = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    except cv2.error:
        pass
    return squared
