import cv2 as cv
from PIL import Image
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def elastic_transformations(alpha, sigma, rng=np.random.RandomState(42),
                            interpolation_order=1, color=False):
    """Returns a function to elastically transform multiple images."""

    # Good values for:
    #   alpha: 2000
    #   sigma: between 40 and 60

    def _elastic_transform(image):
        """`image` is a numpy array of shape (3, W, H) of 3 RGB channels of size W*H."""
        if not image.shape[0] == 3:
            image = image.transpose(2, 0, 1)
        # Take measurements
        image_shape = image[0].shape
        # Make random fields
        dx = rng.uniform(-1, 1, image_shape) * alpha
        dy = rng.uniform(-1, 1, image_shape) * alpha
        # Smooth dx and dy
        sdx = gaussian_filter(dx, sigma=sigma, mode='reflect')
        sdy = gaussian_filter(dy, sigma=sigma, mode='reflect')
        # Make meshgrid
        x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
        # Distort meshgrid indices
        distorted_indices = (y + sdy).reshape(-1, 1), \
                            (x + sdx).reshape(-1, 1)

        # Map cooordinates from image to distorted index set
        transformed_channels = [map_coordinates(channel, distorted_indices, mode='reflect',
                                                order=interpolation_order).reshape(image_shape)
                                for channel in image]
        transformed_image = np.concatenate((transformed_channels[0][:, :, None],
                                            transformed_channels[1][:, :, None],
                                            transformed_channels[2][:, :, None]), axis=2)
        return transformed_image

    return _elastic_transform


def show_img(img: np.ndarray):
    img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    img = img.convert("RGB")
    img.show()
