import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import dataset
import util
import llr
import sr3


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


def plot_random_result():
    path = "dataset/test/"

    file_names = os.listdir(path)
    file_name = random.choice(file_names)
    i = random.choice(range(256))

    current_data_matrix, current_label_matrix = dataset.create_single_file_dataset(
        os.path.join(path, file_name)
    )

    current_data_matrix = current_data_matrix[i : i + 1]
    current_label_matrix = current_label_matrix[i : i + 1]

    # Bicubic Interpolation
    prediction_bi = util.bicubic_interpolation(current_data_matrix)

    # Ridge regression
    window_size = 10
    stride = 5

    model_path_ua_rr = "models/lr_ua_10.pkl"
    pca_path_ua_rr = "models/pca_lr_ua_10.pkl"
    scaler_path_ua_rr = "models/scaler_lr_ua_10.pkl"

    model_path_va_rr = "models/lr_va_10.pkl"
    pca_path_va_rr = "models/pca_lr_va_10.pkl"
    scaler_path_va_rr = "models/scaler_lr_va_10.pkl"

    prediction_rr = llr.predict(
        data_matrix=current_data_matrix,
        model_path_ua=model_path_ua_rr,
        pca_path_ua=pca_path_ua_rr,
        scaler_path_ua=scaler_path_ua_rr,
        model_path_va=model_path_va_rr,
        pca_path_va=pca_path_va_rr,
        scaler_path_va=scaler_path_va_rr,
        window_size=window_size,
        stride=stride,
    )

    # Random Forest
    window_size = 10
    stride = 5

    model_path_ua_rf = "models/lr_ua_10.pkl"
    pca_path_ua_rf = "models/pca_lr_ua_10.pkl"
    scaler_path_ua_rf = "models/scaler_lr_ua_10.pkl"

    model_path_va_rf = "models/lr_va_10.pkl"
    pca_path_va_rf = "models/pca_lr_va_10.pkl"
    scaler_path_va_rf = "models/scaler_lr_va_10.pkl"

    prediction_rf = llr.predict(
        data_matrix=current_data_matrix,
        model_path_ua=model_path_ua_rf,
        pca_path_ua=pca_path_ua_rf,
        scaler_path_ua=scaler_path_ua_rf,
        model_path_va=model_path_va_rf,
        pca_path_va=pca_path_va_rf,
        scaler_path_va=scaler_path_va_rf,
        window_size=window_size,
        stride=stride,
    )

    # SR3 (Regression)
    device = "cpu"
    num_features = 256
    model_path = "models/regression_sr3_24.pth"

    sr3_model = sr3.RegressionSR3(
        device=device,
        num_features=num_features,
        model_path=model_path,
    )

    current_data_matrix_sr3 = util.bicubic_interpolation(current_data_matrix)
    current_data_matrix_sr3 = current_data_matrix_sr3.astype(np.float32)

    x = torch.from_numpy(current_data_matrix_sr3)

    prediction_reg_sr3 = sr3_model.inference(x).detach().numpy()

    # SR3 (Diffusion)
    device = "cuda"
    num_features = 256
    model_path = "models/diffusion_sr3_24.pth"

    sr3_model = sr3.DiffusionSR3(
        device=device,
        T=600,
        num_features=num_features,
        model_path=model_path,
    )

    prediction_diff_sr3 = sr3_model.inference(x).cpu().detach().numpy()

    fig, axs = plt.subplots(2, 7, figsize=(12, 3), constrained_layout=True, dpi=1200)

    # Ground Truth
    axs[0, 0].set_title("HR")
    axs[0, 0].imshow(current_label_matrix[0, 0, :, :])
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[0, 0].set_ylabel("Westward (ua)")

    axs[1, 0].imshow(current_label_matrix[0, 1, :, :])
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    axs[1, 0].set_ylabel("Southward (va)")

    # Input
    axs[0, 1].set_title("LR")
    axs[0, 1].imshow(current_data_matrix[0, 0, :, :])
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    axs[1, 1].imshow(current_data_matrix[0, 1, :, :])
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])

    # Bicubic
    axs[0, 2].set_title("Bicubic")
    axs[0, 2].imshow(prediction_bi[0, 0, :, :])
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])

    axs[1, 2].imshow(prediction_bi[0, 1, :, :])
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])

    # Ridge Regression
    axs[0, 3].set_title("Ridge Regression")
    axs[0, 3].imshow(prediction_rr[0, 0, :, :])
    axs[0, 3].set_xticks([])
    axs[0, 3].set_yticks([])

    axs[1, 3].imshow(prediction_rr[0, 1, :, :])
    axs[1, 3].set_xticks([])
    axs[1, 3].set_yticks([])

    # Random Forest
    axs[0, 4].set_title("Random Forest")
    axs[0, 4].imshow(prediction_rf[0, 0, :, :])
    axs[0, 4].set_xticks([])
    axs[0, 4].set_yticks([])

    axs[1, 4].imshow(prediction_rf[0, 1, :, :])
    axs[1, 4].set_xticks([])
    axs[1, 4].set_yticks([])

    # SR3 (Regression)
    axs[0, 5].set_title("SR3 (Regression)")
    axs[0, 5].imshow(prediction_reg_sr3[0, 0, :, :])
    axs[0, 5].set_xticks([])
    axs[0, 5].set_yticks([])

    axs[1, 5].imshow(prediction_reg_sr3[0, 1, :, :])
    axs[1, 5].set_xticks([])
    axs[1, 5].set_yticks([])

    # SR3 (Diffusion)
    axs[0, 6].set_title("SR3 (Diffusion)")
    axs[0, 6].imshow(prediction_diff_sr3[0, 0, :, :])
    axs[0, 6].set_xticks([])
    axs[0, 6].set_yticks([])

    axs[1, 6].imshow(prediction_diff_sr3[0, 1, :, :])
    axs[1, 6].set_xticks([])
    axs[1, 6].set_yticks([])

    fig.savefig("figures/results_sample.png")
    plt.show()

    return current_data_matrix, current_label_matrix, prediction_bi, prediction_rr, prediction_rf, prediction_reg_sr3, prediction_diff_sr3

# plot_random_sample("dataset/val/")
# plot_random_result()
