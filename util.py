import numpy as np

from PIL import Image


def extract_patches(raw_data: np.ndarray):
    """Converts the raw NREL Wind Toolkit data into patches of
    size 100 by 100 pixels.

    Args:
        raw_data (np.ndarray): Array of the raw data of shape (2, 1600, 1600).

    Returns:
        tuple: ua and va componenets reprsented as 100 by 100
        pixels patches.
    """

    ua = raw_data[0, :, :]
    va = raw_data[1, :, :]

    ua_patches = ua.reshape(16, 100, 16, 100).swapaxes(1, 2).reshape(-1, 100, 100)
    va_patches = va.reshape(16, 100, 16, 100).swapaxes(1, 2).reshape(-1, 100, 100)

    ua_patches = (ua_patches.swapaxes(0, -1) - ua_patches.min(axis=(1, 2))) / (
        ua_patches.max(axis=(1, 2)) - ua_patches.min(axis=(1, 2))
    )
    va_patches = (va_patches.swapaxes(0, -1) - va_patches.min(axis=(1, 2))) / (
        va_patches.max(axis=(1, 2)) - va_patches.min(axis=(1, 2))
    )

    ua_patches = ua_patches.swapaxes(0, -1)
    va_patches = va_patches.swapaxes(0, -1)

    return ua_patches, va_patches


def downsample(patches: np.ndarray):
    """Performs downsampling on the given input patches using array slicing.

    Args:
        patches (np.ndarray): Array of size (batch_size, 100, 100) which
        to be downsampled to an array of size (batch_size, 20, 20).

    Returns:
        np.ndarray: array of the downsampled patches.
    """
    return patches[:, ::5, ::5]


def bicubic_interpolation(patches: np.ndarray):
    """Performs upsampling on the given input patches using bicubic interpolation.

    Args:
        patches (np.ndarray): Array of size (batch_size, 2, 20, 20) which
        to be upsampled using bicubic interpolation to an array of size
        (batch_size, 2, 100, 100).

    Returns:
        np.ndarray: array of the upsampled patches.
    """

    batch_size = patches.shape[0]

    upsampled_patches = np.zeros((batch_size, 2, 100, 100))

    for channel_idx in [0, 1]:
        for i in range(batch_size):
            upsampled_patches[i, channel_idx, :, :] = np.array(
                Image.fromarray(patches[i, channel_idx, :, :]).resize(
                    (100, 100), Image.BICUBIC
                )
            )

    return upsampled_patches


def reconstruct_blocks(blocks: np.ndarray, stride: int):
    """Reconstruct an overlapping block-wise array into a single
    array, where the overlapping segments are averaged.

    Args:
        blocks (np.ndarray): array of shape (num_rows, num_cols, window_size, window_size)
        stride (int): Stride of the rolling window used on the input blocks.

    Returns:
        np.ndarray: array of shape (100, 100)
    """
    window_size = blocks.shape[2]

    reconstructed_matrix = np.zeros((100, 100), dtype=float)
    count_matrix = np.zeros((100, 100), dtype=float)

    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            block = blocks[i, j]
            reconstructed_matrix[
                i * stride : i * stride + window_size,
                j * stride : j * stride + window_size,
            ] += block
            count_matrix[
                i * stride : i * stride + window_size,
                j * stride : j * stride + window_size,
            ] += 1

    reconstructed_matrix /= count_matrix

    return reconstructed_matrix
