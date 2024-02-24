import os
import random
import numpy as np

from PIL import Image


def create_subsampled_dataset(path: str, sample_size: int = None):
    """Creates a data and label tensors from the raw NREL
    Wind Toolkit data.

    Args:
        path (str): Path to the dataset.
        sample_size (int): Number of files to select from the original
        dataset. Defualts to the whole dataset if no sample size is specified.

    Returns:
        tuple: Data tensor of shape (256 * sample_size, 2, 20, 20) and
        a label tensor of shape (256 * sample_size, 2, 100, 100).
    """

    file_names = os.listdir(path)
    file_names_samples = (
        file_names if sample_size == None else random.sample(file_names, sample_size)
    )

    if sample_size == None:
        sample_size = len(file_names)

    data_matrix = np.zeros((256 * sample_size, 2, 20, 20))
    label_matrix = np.zeros((256 * sample_size, 2, 100, 100))

    for index, file in enumerate(file_names_samples):
        raw_data = np.load(os.path.join(path, file))

        ua_patches, uv_patches = extract_patches(raw_data)

        ua_patches_downsampled = downsample(ua_patches)
        va_patches_downsampled = downsample(uv_patches)

        data_matrix[(index * 256) : (index * 256 + 256), 0, :, :] = (
            ua_patches_downsampled
        )
        data_matrix[(index * 256) : (index * 256 + 256), 1, :, :] = (
            va_patches_downsampled
        )

        label_matrix[(index * 256) : (index * 256 + 256), 0, :, :] = ua_patches
        label_matrix[(index * 256) : (index * 256 + 256), 1, :, :] = uv_patches

    return data_matrix, label_matrix


def create_single_file_dataset(path: str):
    """Creates a data and label tensors from the raw NREL
    Wind Toolkit data of a single file (Used mostly at inference time).

    Args:
        path (str): Path to the dataset.

    Returns:
        tuple: Data tensor of shape (256 * sample_size, 2, 20, 20) and
        a label tensor of shape (256 * sample_size, 2, 100, 100).
    """

    data_matrix = np.zeros((256, 2, 20, 20))
    label_matrix = np.zeros((256, 2, 100, 100))

    raw_data = np.load(path)

    ua_patches, uv_patches = extract_patches(raw_data)

    ua_patches_downsampled = downsample(ua_patches)
    va_patches_downsampled = downsample(uv_patches)

    data_matrix[:, 0, :, :] = ua_patches_downsampled
    data_matrix[:, 1, :, :] = va_patches_downsampled

    label_matrix[:, 0, :, :] = ua_patches
    label_matrix[:, 1, :, :] = uv_patches

    return data_matrix, label_matrix


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
