import os

import numpy as np
import tensorflow as tf
import argparse
import json

from PIL import Image

def load_image_into_numpy_array(img):
    (im_width, im_height) = img.size
    return np.array(img.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)