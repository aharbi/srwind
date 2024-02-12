import numpy as np


def extract_patches(path: str):
    """Converts the raw NREL Wind Toolkit data into patches of
    size 100 by 100 pixels.

    Args:
        path (str): Path to the raw data.

    Returns:
        tuple: ua and va componenets reprsented as 100 by 100
        pixels patches.
    """

    wind_profile = np.load(path)

    ua = wind_profile[0, :, :]
    va = wind_profile[1, :, :]

    ua_patches = ua.reshape(16, 100, 16, 100).swapaxes(1, 2).reshape(-1, 100, 100)
    va_patches = va.reshape(16, 100, 16, 100).swapaxes(1, 2).reshape(-1, 100, 100)

    return ua_patches, va_patches
