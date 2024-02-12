import os
import random
import numpy as np
import matplotlib.pyplot as plt


def plot_random_sample(path: str):
    """Plots a random sample from the NREL Wind Toolkit dataset. The plot
    shows both westward (ua) and southward (va) wind componenets along with
    a sample of the high and low resolution measurements.

    Args:
        path (str): Path to the dataset
    """

    file = random.choice(os.listdir(path))
    wind_profile = np.load(os.path.join(path, file))

    wind_profile_HR_sample = wind_profile[0, 750:850, 750:850]
    wind_profile_LR_sample = wind_profile_HR_sample[::5, ::5]

    fig, axs = plt.subplots(2, 2, dpi=300)

    # westward (ua) wind component sample
    axs[0, 0].imshow(wind_profile[0, :, :])
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])

    # southward (va) wind component sample
    axs[0, 1].imshow(wind_profile[1, :, :])
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    # High-resolution westward (ua) sample
    axs[1, 0].imshow(wind_profile_HR_sample)
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])

    # Low-resolution westward (ua) sample
    axs[1, 1].imshow(wind_profile_LR_sample)
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])

    fig.set_size_inches(6.75, 6.75)
    fig.tight_layout()

    fig.savefig("figures/data_sample.png")
    plt.show()


plot_random_sample("dataset/val/")
