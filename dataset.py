import os
import random
import util
import numpy as np

from torch.utils.data import Dataset


class WindDataset(Dataset):
    def __init__(self, data_matrix_path: str, label_matrix_path: str):
        self.data_matrix = np.load(data_matrix_path)
        self.label_matrix = np.load(label_matrix_path)

        assert len(self.data_matrix) == len(self.label_matrix)

    def __len__(self):
        return len(self.data_matrix)

    def __getitem__(self, idx):
        LR = self.data_matrix[idx, :, :, :]
        HR = self.label_matrix[idx, :, :, :]

        LR = np.expand_dims(LR, 0)

        LR_upsampled = util.bicubic_interpolation(LR)
        LR_upsampled = LR_upsampled[0, :, :, :]

        LR_upsampled = LR_upsampled.swapaxes(0, 1).swapaxes(1, -1)
        HR = HR.swapaxes(0, 1).swapaxes(1, -1)

        return LR_upsampled, HR


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

        ua_patches, va_patches = util.extract_patches(raw_data)

        ua_patches_downsampled = util.downsample(ua_patches)
        va_patches_downsampled = util.downsample(va_patches)

        data_matrix[(index * 256) : (index * 256 + 256), 0, :, :] = (
            ua_patches_downsampled
        )
        data_matrix[(index * 256) : (index * 256 + 256), 1, :, :] = (
            va_patches_downsampled
        )

        label_matrix[(index * 256) : (index * 256 + 256), 0, :, :] = ua_patches
        label_matrix[(index * 256) : (index * 256 + 256), 1, :, :] = va_patches

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

    ua_patches, va_patches = util.extract_patches(raw_data)

    ua_patches_downsampled = util.downsample(ua_patches)
    va_patches_downsampled = util.downsample(va_patches)

    data_matrix[:, 0, :, :] = ua_patches_downsampled
    data_matrix[:, 1, :, :] = va_patches_downsampled

    label_matrix[:, 0, :, :] = ua_patches
    label_matrix[:, 1, :, :] = va_patches

    return data_matrix, label_matrix


def generate_random_dataset(dataset_path: str, save_path: str, size: int):

    file_names = os.listdir(dataset_path)

    data_matrix = np.zeros((size, 2, 20, 20))
    label_matrix = np.zeros((size, 2, 100, 100))

    for i in range(size):

        current_file = random.choice(file_names)

        raw_data = np.load(os.path.join(dataset_path, current_file))

        idx_x = random.choice(range(1500))
        idx_y = random.choice(range(1500))

        ua_patch = raw_data[0, idx_x : (idx_x + 100), idx_y : (idx_y + 100)]
        va_patch = raw_data[1, idx_x : (idx_x + 100), idx_y : (idx_y + 100)]

        ua_patch = (ua_patch - ua_patch.min()) / (ua_patch.max() - ua_patch.min())
        va_patch = (va_patch - va_patch.min()) / (va_patch.max() - va_patch.min())

        ua_patch_downsampled = ua_patch[::5, ::5]
        va_patch_downsampled = va_patch[::5, ::5]

        data_matrix[i, 0, :, :] = ua_patch_downsampled
        data_matrix[i, 1, :, :] = va_patch_downsampled

        label_matrix[i, 0, :, :] = ua_patch
        label_matrix[i, 1, :, :] = va_patch

    with open(save_path + "data_matrix.npy", "wb") as f:
        np.save(f, data_matrix)

    with open(save_path + "label_matrix.npy", "wb") as f:
        np.save(f, label_matrix)

    return data_matrix, label_matrix
