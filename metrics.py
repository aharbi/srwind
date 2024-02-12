import numpy as np


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
    psnr = 20 * np.log10(np.max(ref_patch) / rmse)
    return 100 if (np.isnan(psnr) or np.isinf(psnr)) else psnr


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
    return None


def kinetic_energy_spectra(patch: np.ndarray):
    return None


def compute_metrics(ref_patches: np.ndarray, HR_patches: np.ndarray):
    """Computes the performance metrics given the reference HR patches and
    the predicted HR patches.

    Args:
        ref_patches (np.ndarray): Array of size (batch_size, 2, 100, 100)
        HR_patches (np.ndarray): Array of size (batch_size, 2, 100, 100)

    Returns:
        tuple: computed metrics as a tuple (PSNR, MSE, MAE, SSIM)
    """

    psnr_log = []
    mse_log = []
    mae_log = []
    ssim_log = []

    n = ref_patches.shape[0]
    for channel_idx in [0, 1]:
        for i in range(n):
            psnr_log.append(
                peak_signal_to_noise_ratio(
                    ref_patches[i, channel_idx, :, :], HR_patches[i, channel_idx, :, :]
                )
            )
            mse_log.append(
                mean_squared_error(
                    ref_patches[i, channel_idx, :, :], HR_patches[i, channel_idx, :, :]
                )
            )
            mae_log.append(
                mean_absolute_error(
                    ref_patches[i, channel_idx, :, :], HR_patches[i, channel_idx, :, :]
                )
            )

    avg_psnr = np.mean(psnr_log)
    avg_mse = np.mean(mse_log)
    avg_mae = np.mean(mae_log)
    avg_ssim = 0

    return avg_psnr, avg_mse, avg_mae, avg_ssim
