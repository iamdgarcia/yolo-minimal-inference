import numpy as np

def resize(image: np.ndarray, new_shape: tuple) -> np.ndarray:
    """
    Resize an image or a batch of images to a new shape using NumPy interpolation.

    Args:
        image (np.ndarray): Input image as a NumPy array. Can be (H, W, C) or (B, H, W, C).
        new_shape (tuple): Desired dimensions (new_height, new_width).

    Returns:
        np.ndarray: Resized image(s) as a NumPy array with the same number of dimensions.
    """
    is_batch = image.ndim == 4  # Check if batch dimension is present
    if is_batch:
        batch_size = image.shape[0]
        images = image
    else:
        batch_size = 1
        images = image[np.newaxis, ...]  # Add batch dimension for uniform processing

    old_height, old_width = images.shape[1:3]
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

    # Apply interpolation grid to each image in the batch
    resized_images = images[:, row_indices[:, None], col_indices, :]

    return resized_images if is_batch else resized_images[0]  # Remove batch dim if input was a single image
