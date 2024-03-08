import os
import numpy as np
import torch
import util
import dataset
import llr
import sr3

from skimage.metrics import structural_similarity as ssim


def peak_signal_to_noise_ratio(ref_patch: np.ndarray, HR_patch: np.ndarray):
    """Computes the peak signal-to-noise ratio between a reference high-resolution
    patch and a predicted high-resolution patch.

    Args:
        ref_patch (np.ndarray): Reference high-resolution patch.
        HR_patch (np.ndarray): Predicted high-resolution patch.

    Returns:
        float: The peak signal-to-noise ratio (in dB)
    """
    rmse = 1 / 100 * np.linalg.norm(ref_patch.flatten() - HR_patch.flatten(), 2)
    if rmse != 0:
        psnr = 20 * np.log10(1 / rmse)
        return psnr
    else:
        return float("inf")


def mean_squared_error(ref_patch: np.ndarray, HR_patch: np.ndarray):
    """Computes the mean squared error between a reference high-resolution
    patch and a predicted high-resolution patch.

    Args:
        ref_patch (np.ndarray): Reference high-resolution patch.
        HR_patch (np.ndarray): Predicted high-resolution patch.

    Returns:
        float: The mean squared error.
    """
    return (1 / (100 * 100)) * np.linalg.norm(
        ref_patch.flatten() - HR_patch.flatten(), 2
    ) ** 2


def mean_absolute_error(ref_patch: np.ndarray, HR_patch: np.ndarray):
    """Computes the mean absolute error between a reference high-resolution
    patch and a predicted high-resolution patch.

    Args:
        ref_patch (np.ndarray): Reference high-resolution patch.
        HR_patch (np.ndarray): Predicted high-resolution patch.

    Returns:
        float: The mean absolute error.
    """
    return (1 / (100 * 100)) * np.linalg.norm(
        ref_patch.flatten() - HR_patch.flatten(), 1
    )


def structural_similarity_index(ref_patch: np.ndarray, HR_patch: np.ndarray):
    """Computes the structural similarity index measure (SSIM) between a reference
    high-resolution patch and a predicted high-resolution patch.

    Args:
        ref_patch (np.ndarray): Reference high-resolution patch.
        HR_patch (np.ndarray): Predicted high-resolution patch.

    Returns:
        float: The structural similarity index.
    """
    return ssim(ref_patch, HR_patch, data_range=1)


def kinetic_energy_spectra(patch: np.ndarray):
    return None


def compute_metrics(ref_patches: np.ndarray, HR_patches: np.ndarray):
    """Computes the performance metrics per patch given the reference HR patches and
    the predicted HR patches.

    Args:
        ref_patches (np.ndarray): Array of size (batch_size, 2, 100, 100)
        HR_patches (np.ndarray): Array of size (batch_size, 2, 100, 100)

    Returns:
        np.ndarray: computed metrics as an array with batch_size rows and
        two columns representing the wind componenets with a depth of four
        representing the computed metrics (PSNR, MSE, MAE, SSIM).
    """

    psnr_log_ua = []
    psnr_log_va = []

    mse_log_ua = []
    mse_log_va = []

    mae_log_ua = []
    mae_log_va = []

    ssim_log_ua = []
    ssim_log_va = []

    n = ref_patches.shape[0]
    for i in range(n):
        psnr_log_ua.append(
            peak_signal_to_noise_ratio(ref_patches[i, 0, :, :], HR_patches[i, 0, :, :])
        )
        mse_log_ua.append(
            mean_squared_error(ref_patches[i, 0, :, :], HR_patches[i, 0, :, :])
        )
        mae_log_ua.append(
            mean_absolute_error(ref_patches[i, 0, :, :], HR_patches[i, 0, :, :])
        )
        ssim_log_ua.append(
            structural_similarity_index(ref_patches[i, 0, :, :], HR_patches[i, 0, :, :])
        )

        psnr_log_va.append(
            peak_signal_to_noise_ratio(ref_patches[i, 1, :, :], HR_patches[i, 1, :, :])
        )
        mse_log_va.append(
            mean_squared_error(ref_patches[i, 1, :, :], HR_patches[i, 1, :, :])
        )
        mae_log_va.append(
            mean_absolute_error(ref_patches[i, 1, :, :], HR_patches[i, 1, :, :])
        )
        ssim_log_va.append(
            structural_similarity_index(ref_patches[i, 1, :, :], HR_patches[i, 1, :, :])
        )

    metrics_array = np.zeros((n, 2, 4))

    metrics_array[:, 0, 0] = psnr_log_ua
    metrics_array[:, 0, 1] = mse_log_ua
    metrics_array[:, 0, 2] = mae_log_ua
    metrics_array[:, 0, 3] = ssim_log_ua

    metrics_array[:, 1, 0] = psnr_log_va
    metrics_array[:, 1, 1] = mse_log_va
    metrics_array[:, 1, 2] = mae_log_va
    metrics_array[:, 1, 3] = ssim_log_va

    return metrics_array


def compute_metrics_llr(
    path: str,
    model_path_ua,
    pca_path_ua,
    scaler_path_ua,
    model_path_va,
    pca_path_va,
    scaler_path_va,
    window_size,
    stride,
):

    file_names = os.listdir(path)
    n = len(file_names)

    metrics_array = np.zeros((256 * n, 2, 4))

    for i, file_name in enumerate(file_names):
        current_data_matrix, current_label_matrix = dataset.create_single_file_dataset(
            os.path.join(path, file_name)
        )

        prediction_lr = llr.predict(
            data_matrix=current_data_matrix,
            model_path_ua=model_path_ua,
            pca_path_ua=pca_path_ua,
            scaler_path_ua=scaler_path_ua,
            model_path_va=model_path_va,
            pca_path_va=pca_path_va,
            scaler_path_va=scaler_path_va,
            window_size=window_size,
            stride=stride,
        )

        metrics_array[i * 256 : i * 256 + 256, :, :] = compute_metrics(
            current_label_matrix, prediction_lr
        )

        print("Current Iteration (LLR): {} / {}".format(i, n))

    return metrics_array


def compute_metrics_regression_sr3(
    path: str,
    model_path: str,
    device: str = "cuda",
    num_features: int = 256,
):

    sr3_model = sr3.RegressionSR3(
        device=device,
        num_features=num_features,
        model_path=model_path,
    )

    file_names = os.listdir(path)
    n = len(file_names)

    metrics_array = np.zeros((256 * n, 2, 4))

    for i, file_name in enumerate(file_names):
        current_data_matrix, current_label_matrix = dataset.create_single_file_dataset(
            os.path.join(path, file_name)
        )

        current_data_matrix = util.bicubic_interpolation(current_data_matrix)
        current_data_matrix = current_data_matrix.astype(np.float32)

        x = torch.from_numpy(current_data_matrix)

        prediction_sr3 = sr3_model.inference(x).detach().numpy()

        metrics_array[i * 256 : i * 256 + 256, :, :] = compute_metrics(
            current_label_matrix, prediction_sr3
        )

        print("Current Iteration (SR3): {} / {}".format(i, n))

    return metrics_array


def compute_metrics_diffusion_sr3(
    path: str,
    model_path: str,
    device: str = "cuda",
    T: int = 500,
    num_features: int = 256,
):

    sr3_model = sr3.DiffusionSR3(
        device=device,
        T=T,
        num_features=num_features,
        model_path=model_path,
    )

    file_names = os.listdir(path)
    sample_size = 2

    metrics_array = np.zeros((256 * sample_size, 2, 4))

    for i, file_name in enumerate(file_names[:sample_size]):
        current_data_matrix, current_label_matrix = dataset.create_single_file_dataset(
            os.path.join(path, file_name)
        )

        current_data_matrix = util.bicubic_interpolation(current_data_matrix)
        current_data_matrix = current_data_matrix.astype(np.float32)

        for j in range(0, 256, 2):
            x = torch.from_numpy(current_data_matrix[j : j + 2])

            prediction_sr3 = sr3_model.inference(x).detach().numpy()

            metrics_array[i * 256 + j : i * 256 + j + 2, :, :] = compute_metrics(
                current_label_matrix[j : j + 2], prediction_sr3
            )

            print("Current Sub-iteration: {} / {}".format(j, 256))

        print("Current Iteration (SR3): {} / {}".format(i, sample_size))

    return metrics_array


def compute_metrics_bicubic(path: str):

    file_names = os.listdir(path)
    n = len(file_names)

    metrics_array = np.zeros((256 * n, 2, 4))

    for i, file_name in enumerate(file_names):
        current_data_matrix, current_label_matrix = dataset.create_single_file_dataset(
            os.path.join(path, file_name)
        )

        prediction_bi = util.bicubic_interpolation(current_data_matrix)

        metrics_array[i * 256 : i * 256 + 256, :, :] = compute_metrics(
            current_label_matrix, prediction_bi
        )

        print("Current Iteration (Bicubic): {} / {}".format(i, n))

    return metrics_array
