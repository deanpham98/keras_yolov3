from __future__ import division

import pdb

from keras import backend as K
import numpy as np
import cv2
from PIL import Image

from .transform import change_transform_origin, transform_aabb


def read_image_bgr(path):
    try:
         image = np.asarray(Image.open(path).convert('RGB'))
    except:
         pdb.set_trace()
    return image[:, :, ::-1].copy() 


def adjust_transform_for_image(transform, image, relative_translation):
    """ Adjust a transformation for a specific image.
    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    """
    height, width, channels = image.shape

    result = transform

    # Scale the translation with the image size if specified.
    if relative_translation:
        result[0:2, 2] *= [width, height]

    # Move the origin of transformation.
    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    return result


class TransformParameters:
    """ Struct holding parameters determining how to apply a transformation to an image.
    # Arguments
        fill_mode:             One of: 'constant', 'nearest', 'reflect', 'wrap'
        interpolation:         One of: 'nearest', 'linear', 'cubic', 'area', 'lanczos4'
        cval:                  Fill value to use with fill_mode='constant'
        data_format:           Same as for keras.preprocessing.image.apply_transform
        relative_translation:  If true (the default), interpret translation as a factor of the image size.
                               If false, interpret it as absolute pixels.
    """
    def __init__(
        self,
        fill_mode            = 'nearest',
        interpolation        = 'linear',
        cval                 = 0,
        data_format          = None,
        relative_translation = True,
    ):
        self.fill_mode            = fill_mode
        self.cval                 = cval
        self.interpolation        = interpolation
        self.relative_translation = relative_translation

        if data_format is None:
            data_format = K.image_data_format()
        self.data_format = data_format

        if data_format == 'channels_first':
            self.channel_axis = 0
        elif data_format == 'channels_last':
            self.channel_axis = 2
        else:
            raise ValueError("invalid data_format, expected 'channels_first' or 'channels_last', got '{}'".format(data_format))

    def cvBorderMode(self):
        if self.fill_mode == 'constant':
            return cv2.BORDER_CONSTANT
        if self.fill_mode == 'nearest':
            return cv2.BORDER_REPLICATE
        if self.fill_mode == 'reflect':
            return cv2.BORDER_REFLECT_101
        if self.fill_mode == 'wrap':
            return cv2.BORDER_WRAP

    def cvInterpolation(self):
        if self.interpolation == 'nearest':
            return cv2.INTER_NEAREST
        if self.interpolation == 'linear':
            return cv2.INTER_LINEAR
        if self.interpolation == 'cubic':
            return cv2.INTER_CUBIC
        if self.interpolation == 'area':
            return cv2.INTER_AREA
        if self.interpolation == 'lanczos4':
            return cv2.INTER_LANCZOS4


def apply_transform(matrix, image, params):
    """
    Apply a transformation to an image.
    The origin of transformation is at the top left corner of the image.
    The matrix is interpreted such that a point (x, y) on the original image is moved to transform * (x, y) in the generated image.
    Mathematically speaking, that means that the matrix is a transformation from the transformed image space to the original image space.
    Parameters:
      matrix: A homogeneous 3 by 3 matrix holding representing the transformation to apply.
      image:  The image to transform.
      params: The transform parameters (see TransformParameters)
    """
    if params.channel_axis != 2:
        image = np.moveaxis(image, params.channel_axis, 2)

    output = cv2.warpAffine(
        image,
        matrix[:2, :],
        dsize       = (image.shape[1], image.shape[0]),
        flags       = params.cvInterpolation(),
        borderMode  = params.cvBorderMode(),
        borderValue = params.cval,
    )

    if params.channel_axis != 2:
        output = np.moveaxis(output, 2, params.channel_axis)
    return output


def resize_image(img, annotations, input_shape=(416, 416)):
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    img_w, img_h = img.size
    w, h = input_shape
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = img.resize((new_w,new_h), Image.BICUBIC)
    boxed_image = Image.new('RGB', input_shape, (128,128,128))
    boxed_image.paste(resized_image, ((w-new_w)//2,(h-new_h)//2))
    img_size = np.array(img.size)
    input_size = np.array(input_shape[::-1])
    new_size = (img_size * np.min(input_size/img_size)).astype('int32')
    annotations[:, 0:2] = (annotations[:, 0:2] * new_size/img_size + (input_size - new_size)/2)
    annotations[:, 2:4] = (annotations[:, 2:4] * new_size/img_size + (input_size - new_size)/2)
    filter = (annotations[:, 2] - annotations[:, 0] > 3) * (annotations[:, 3] - annotations[:, 1] > 3)
    annotations = annotations[filter]

    return np.array(boxed_image), annotations.astype('int32')


