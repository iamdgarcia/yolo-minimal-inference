import numpy as np

def resize(image, new_shape):
        """
        Resize image to new_shape using NumPy.
        """
        old_height, old_width = image.shape[:2]
        new_height, new_width = new_shape
        row_scale = new_height / old_height
        col_scale = new_width / old_width

        # Create grid for interpolation
        row_indices = (np.arange(new_height) / row_scale).astype(int)
        col_indices = (np.arange(new_width) / col_scale).astype(int)

        # Apply the interpolation grid
        resized = image[row_indices[:, None], col_indices, :]
        return resized