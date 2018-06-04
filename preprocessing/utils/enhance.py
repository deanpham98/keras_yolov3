import numpy as np
import cv2
from PIL import Image, ImageEnhance

def array_to_img(x):
	return Image.fromarray(x[:,:,::-1])

def img_to_array(img):
	return np.array(img)[:,:,::-1]

def apply_brightness_shift(x, brightness):
	x = array_to_img(x)
	x = ImageEnhance.Brightness(x).enhance(brightness)
	x = img_to_array(x)
	return x

def apply_contrast_shift(x, contrast):
	x = array_to_img(x)
	x = ImageEnhance.Contrast(x).enhance(contrast)
	x = img_to_array(x)
	return x

def apply_sharpness_shift(x, sharpness):
	x = array_to_img(x)
	x = ImageEnhance.Sharpness(x).enhance(sharpness)
	x = img_to_array(x)
	return x


class random_enhance_generator():
	def __init__ (self,
			 	  brightness_range=(0.2, 1.8),
			 	  contrast_range=(0.6, 0.8),
			 	  sharpness_range=(0.6, 0.8)
 		):
		self.brightness_range = brightness_range
		self.contrast_range = contrast_range
		self.sharpness_range = sharpness_range

	def random_brightness(self, x):
	    """Performs a random brightness shift.
	    # Arguments
	        x: Input tensor. Must be 3D.
	        brightness_range: Tuple of floats; brightness range.
	        channel_axis: Index of axis for channels in the input tensor.
	    # Returns
	        Numpy image tensor.
	    # Raises
	        ValueError if `brightness_range` isn't a tuple.
	    """
	    if len(self.brightness_range) != 2:
	        raise ValueError(
	            '`brightness_range should be tuple or list of two floats. '
	            'Received: %s' % self.brightness_range)

	    u = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
	    return apply_brightness_shift(x, u)

	def random_contrast(self, x):
	    """Performs a random contrast shift.
	    # Arguments
	        x: Input tensor. Must be 3D.
	        contrast_range: Tuple of floats; contrast range.
	        channel_axis: Index of axis for channels in the input tensor.
	    # Returns
	        Numpy image tensor.
	    # Raises
	        ValueError if `contrast_range` isn't a tuple.
	    """
	    if len(self.contrast_range) != 2:
	        raise ValueError(
	            '`contrast_range should be tuple or list of two floats. '
	            'Received: %s' % self.contrast_range)

	    u = np.random.uniform(self.contrast_range[0], self.contrast_range[1])
	    return apply_contrast_shift(x, u)

	def random_sharpness(self, x):
	    """Performs a random sharpness shift.
	    # Arguments
	        x: Input tensor. Must be 3D.
	        sharpness_range: Tuple of floats; sharpness range.
	        channel_axis: Index of axis for channels in the input tensor.
	    # Returns
	        Numpy image tensor.
	    # Raises
	        ValueError if `sharpness_range` isn't a tuple.
	    """
	    if len(self.sharpness_range) != 2:
	        raise ValueError(
	            '`sharpness_range should be tuple or list of two floats. '
	            'Received: %s' % self.sharpness_range)

	    u = np.random.uniform(self.sharpness_range[0], self.sharpness_range[1])
	    return apply_sharpness_shift(x, u)

	def random_enhance(self, x):
		"""
		Performs random enhance for input image x.
		# Arguments
			x: Input tensor. Must be 3D
		# Returns 
			Numpy image tensor
		"""
		x = self.random_brightness(x) if self.brightness_range else x

		x = self.random_contrast(x) if self.contrast_range else x

		x = self.random_sharpness(x) if self.sharpness_range else x 

		return x
