import numpy as np

def resize(image: np.ndarray, new_shape: tuple) -> np.ndarray:
    """
    Resize an image to a new shape using NumPy interpolation.

    Args:
        image (np.ndarray): Input image as a NumPy array (H, W, C).
        new_shape (tuple): Desired dimensions (new_height, new_width).

    Returns:
        np.ndarray: Resized image as a NumPy array.
    """
    old_height, old_width = image.shape[:2]
    new_height, new_width = new_shape

    # Calculate scaling factors
    row_scale = new_height / old_height
    col_scale = new_width / old_width

    # Create grid indices for interpolation
    row_indices = (np.arange(new_height) / row_scale).astype(int)
    col_indices = (np.arange(new_width) / col_scale).astype(int)

    # Clip indices to stay within bounds (to avoid IndexError)
    row_indices = np.clip(row_indices, 0, old_height - 1)
    col_indices = np.clip(col_indices, 0, old_width - 1)

    # Apply the interpolation grid
    resized = image[row_indices[:, None], col_indices, :]
    return resized
