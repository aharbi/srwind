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
    return 20 * np.log10(np.max(ref_patch) / rmse)


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
