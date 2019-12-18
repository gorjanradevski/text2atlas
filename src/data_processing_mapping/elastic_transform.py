import cv2 as cv
from PIL import Image
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


# Elastic transform
def elastic_transformations(alpha, sigma, rng=np.random.RandomState(42), 
                            interpolation_order=1, color=False):
    """Returns a function to elastically transform multiple images."""
    # Good values for:
    #   alpha: 2000
    #   sigma: between 40 and 60
    def _elastic_transform_2D(images):
        """`images` is a numpy array of shape (K, M, N) of K images of size M*N."""
        # Take measurements
        image_shape = images[0].shape
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
        transformed_images = [map_coordinates(image, distorted_indices, mode='reflect',
                                              order=interpolation_order).reshape(image_shape)
                              for image in images]
        return transformed_images
    
    def _elastic_transform_3D(images):
        """`images` is a numpy array of shape (3, M, N) of 3 RGB channels of size M*N images."""
        if not images.shape[0] == 3:
            images = images.transpose(2,0,1)
        # Take measurements
        image_shape = images[0].shape
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
        transformed_images = [map_coordinates(image, distorted_indices, mode='reflect',
                                              order=interpolation_order).reshape(image_shape)
                              for image in images]
        transformed_images = np.concatenate((transformed_images[0][:,:,None],
                                             transformed_images[1][:,:,None],
                                             transformed_images[2][:,:,None]), axis=2)
        return transformed_images
    
    if color:
        return _elastic_transform_3D
    else:
        return _elastic_transform_2D


def show_img(img: np.ndarray):
    IMG = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    IMG = IMG.convert("RGB")
    IMG.show()