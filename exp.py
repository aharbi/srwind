import numpy as np
import util
import dataset
import llr
import sr3
import metrics


def exp_1():
    """Experiment 1: Compares the performance of a ridge regression model
    to bicubic interpolation on the validation.
    """
    metrics_array_rr = metrics.compute_metrics_llr(
        "dataset/val/",
        "models/lr_ua.pkl",
        "models/pca_lr_ua.pkl",
        "models/lr_va.pkl",
        "models/pca_lr_va.pkl",
        20,
        5,
        "results/lr_val_metrics",
    )

    metrics_array_bicubic = metrics.compute_metrics_bicubic(
        "dataset/val/", "results/bicubic_val_metrics"
    )

    psnr_rr = np.vstack([metrics_array_rr[:, 0, 0], metrics_array_rr[:, 1, 0]]).mean()
    mse_rr = np.vstack([metrics_array_rr[:, 0, 1], metrics_array_rr[:, 1, 1]]).mean()
    mae_rr = np.vstack([metrics_array_rr[:, 0, 2], metrics_array_rr[:, 1, 2]]).mean()

    psnr_bicubic = np.vstack(
        [metrics_array_bicubic[:, 0, 0], metrics_array_bicubic[:, 1, 0]]
    ).mean()
    mse_bicubic = np.vstack(
        [metrics_array_bicubic[:, 0, 1], metrics_array_bicubic[:, 1, 1]]
    ).mean()
    mae_bicubic = np.vstack(
        [metrics_array_bicubic[:, 0, 2], metrics_array_bicubic[:, 1, 2]]
    ).mean()

    print("PSNR: ", psnr_rr, psnr_bicubic)
    print("MSE: ", mse_rr, mse_bicubic)
    print("MAE: ", mae_rr, mae_bicubic)


def exp_2():
    """Experiment 2: Compares the performance of a local regression models
    to bicubic interpolation on the testing set.
    """
    # Models parameters
    window_size = 10
    stride = 5
    pca_components = 16
    train = True

    lr_args = {"alpha": 0.5, "fit_intercept": True}

    rf_args = {
        "n_estimators": 1000,
        "max_depth": 16,
        "n_jobs": -1,
        "verbose": 2,
    }

    # Loading data
    if train:
        data_matrix = np.load("dataset/data_matrix.npy")
        label_matrix = np.load("dataset/label_matrix.npy")

        X, Y = llr.generate_features(data_matrix, label_matrix, window_size)

        # Training models
        llr.linear_regression_super_resolution(
            X[0], Y[0], "models/", lr_args, pca_components, name="lr_ua.pkl"
        )

        llr.linear_regression_super_resolution(
            X[1], Y[1], "models/", lr_args, pca_components, name="lr_va.pkl"
        )

        # Train random forest
        llr.random_forest_super_resolution(
            X[0], Y[0], "models/", rf_args, pca_components, name="rfsr_ua.pkl"
        )
        llr.random_forest_super_resolution(
            X[1], Y[1], "models/", rf_args, pca_components, name="rfsr_va.pkl"
        )

    # Ridge regression metrics
    metrics_array_rr = metrics.compute_metrics_llr(
        path="dataset/test/",
        model_path_ua="models/lr_ua.pkl",
        pca_path_ua="models/pca_lr_ua.pkl",
        scaler_path_ua="models/scaler_lr_ua.pkl",
        model_path_va="models/lr_va.pkl",
        pca_path_va="models/pca_lr_va.pkl",
        scaler_path_va="models/scaler_lr_va.pkl",
        window_size=window_size,
        stride=stride,
    )

    psnr_rr = np.vstack([metrics_array_rr[:, 0, 0], metrics_array_rr[:, 1, 0]])
    mse_rr = np.vstack([metrics_array_rr[:, 0, 1], metrics_array_rr[:, 1, 1]])
    mae_rr = np.vstack([metrics_array_rr[:, 0, 2], metrics_array_rr[:, 1, 2]])
    ssim_rr = np.vstack([metrics_array_rr[:, 0, 3], metrics_array_rr[:, 1, 3]])

    # Randfom forest metrics
    metrics_array_rr = metrics.compute_metrics_llr(
        path="dataset/test/",
        model_path_ua="models/rfsr_ua.pkl",
        pca_path_ua="models/pca_rfsr_ua.pkl",
        scaler_path_ua="models/scaler_rfsr_ua.pkl",
        model_path_va="models/rfsr_va.pkl",
        pca_path_va="models/pca_rfsr_va.pkl",
        scaler_path_va="models/scaler_rfsr_va.pkl",
        window_size=window_size,
        stride=stride,
    )

    psnr_rfsr = np.vstack([metrics_array_rr[:, 0, 0], metrics_array_rr[:, 1, 0]])
    mse_rfsr = np.vstack([metrics_array_rr[:, 0, 1], metrics_array_rr[:, 1, 1]])
    mae_rfsr = np.vstack([metrics_array_rr[:, 0, 2], metrics_array_rr[:, 1, 2]])
    ssim_rfsr = np.vstack([metrics_array_rr[:, 0, 3], metrics_array_rr[:, 1, 3]])

    # Bicubic Interpolation metrics
    metrics_array_bicubic = metrics.compute_metrics_bicubic("dataset/test/")

    psnr_bicubic = np.vstack(
        [metrics_array_bicubic[:, 0, 0], metrics_array_bicubic[:, 1, 0]]
    )
    mse_bicubic = np.vstack(
        [metrics_array_bicubic[:, 0, 1], metrics_array_bicubic[:, 1, 1]]
    )
    mae_bicubic = np.vstack(
        [metrics_array_bicubic[:, 0, 2], metrics_array_bicubic[:, 1, 2]]
    )
    ssim_bicubic = np.vstack(
        [metrics_array_bicubic[:, 0, 3], metrics_array_bicubic[:, 1, 3]]
    )

    print("RR - Average PSNR: ", np.ma.masked_invalid(psnr_rr).mean())
    print("RR - Average MSE: ", mse_rr.mean())
    print("RR - Average MAE: ", mae_rr.mean())
    print("RR - Average SSIM: ", ssim_rr.mean())

    print("RR - Standard Deviation of PSNR: ", np.ma.masked_invalid(psnr_rr).std())
    print("RR - Standard Deviation of MSE: ", mse_rr.std())
    print("RR - Standard Deviation of MAE: ", mae_rr.std())
    print("RR - Standard Deviation of SSIM: ", ssim_rr.std())

    print("RFSR - Average PSNR: ", np.ma.masked_invalid(psnr_rfsr).mean())
    print("RFSR - Average MSE: ", mse_rfsr.mean())
    print("RFSR - Average MAE: ", mae_rfsr.mean())
    print("RFSR - Average SSIM: ", ssim_rfsr.mean())

    print("RFSR - Standard Deviation of PSNR: ", np.ma.masked_invalid(psnr_rfsr).std())
    print("RFSR - Standard Deviation of MSE: ", mse_rfsr.std())
    print("RFSR - Standard Deviation of MAE: ", mae_rfsr.std())
    print("RFSR - Standard Deviation of SSIM: ", ssim_rfsr.std())

    print("Bicubic - Average PSNR: ", np.ma.masked_invalid(psnr_bicubic).mean())
    print("Bicubic - Average MSE: ", mse_bicubic.mean())
    print("Bicubic - Average MAE: ", mae_bicubic.mean())
    print("Bicubic - Average SSIM: ", ssim_bicubic.mean())

    print(
        "Bicubic - Standard Deviation of PSNR: ",
        np.ma.masked_invalid(psnr_bicubic).std(),
    )
    print("Bicubic - Standard Deviation of MSE: ", mse_bicubic.std())
    print("Bicubic - Standard Deviation of MAE: ", mae_bicubic.std())
    print("Bicubic - Standard Deviation of SSIM: ", ssim_bicubic.std())


def exp_3():
    device = "cuda"
    num_features = 256
    num_epochs = 25
    batch_size = 32

    regression_SR3 = sr3.RegressionSR3(
        device=device,
        num_features=num_features,
    )

    wind_dataset = dataset.WindDataset(
        "dataset/data_matrix.npy", "dataset/label_matrix.npy"
    )

    regression_SR3.train(
        dataset=wind_dataset,
        save_path="models/",
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

    metrics_array_regression_sr3 = metrics.compute_metrics_regression_sr3(
        "dataset/test/",
        "models/regression_sr3_{}.pth".format(num_epochs - 1),
        device=device,
        num_features=num_features,
    )

    psnr_regression_sr3 = np.vstack(
        [metrics_array_regression_sr3[:, 0, 0], metrics_array_regression_sr3[:, 1, 0]]
    )
    mse_regression_sr3 = np.vstack(
        [metrics_array_regression_sr3[:, 0, 1], metrics_array_regression_sr3[:, 1, 1]]
    )
    mae_regression_sr3 = np.vstack(
        [metrics_array_regression_sr3[:, 0, 2], metrics_array_regression_sr3[:, 1, 2]]
    )
    ssim_regression_sr3 = np.vstack(
        [metrics_array_regression_sr3[:, 0, 3], metrics_array_regression_sr3[:, 1, 3]]
    )

    print(
        "SR3 (Regression) - Average PSNR: ",
        np.ma.masked_invalid(psnr_regression_sr3).mean(),
    )
    print("SR3 (Regression) - Average MSE: ", mse_regression_sr3.mean())
    print("SR3 (Regression) - Average MAE: ", mae_regression_sr3.mean())
    print("SR3 (Regression) - Average SSIM: ", ssim_regression_sr3.mean())

    print(
        "SR3 (Regression) - Standard Deviation of PSNR: ",
        np.ma.masked_invalid(psnr_regression_sr3).std(),
    )
    print("SR3 (Regression) - Standard Deviation of MSE: ", mse_regression_sr3.std())
    print("SR3 (Regression) - Standard Deviation of MAE: ", mae_regression_sr3.std())
    print("SR3 (Regression) - Standard Deviation of SSIM: ", ssim_regression_sr3.std())


def exp_4():
    device = "cuda"
    num_features = 256
    num_epochs = 25
    batch_size = 32
    T = 600

    diffusion_SR3 = sr3.DiffusionSR3(device=device, T=T, num_features=num_features)

    wind_dataset = dataset.WindDataset(
        "dataset/data_matrix.npy", "dataset/label_matrix.npy"
    )

    diffusion_SR3.train(
        dataset=wind_dataset,
        save_path="models/",
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

    metrics_array_diffusion_sr3 = metrics.compute_metrics_diffusion_sr3(
        "dataset/test/",
        "models/diffusion_sr3_{}.pth".format(num_epochs - 1),
        device=device,
        T=T,
        num_features=num_features,
    )

    psnr_diffusion_sr3 = np.vstack(
        [metrics_array_diffusion_sr3[:, 0, 0], metrics_array_diffusion_sr3[:, 1, 0]]
    )
    mse_diffusion_sr3 = np.vstack(
        [metrics_array_diffusion_sr3[:, 0, 1], metrics_array_diffusion_sr3[:, 1, 1]]
    )
    mae_diffusion_sr3 = np.vstack(
        [metrics_array_diffusion_sr3[:, 0, 2], metrics_array_diffusion_sr3[:, 1, 2]]
    )
    ssim_diffusion_sr3 = np.vstack(
        [metrics_array_diffusion_sr3[:, 0, 3], metrics_array_diffusion_sr3[:, 1, 3]]
    )

    print(
        "SR3 (Diffusion) - Average PSNR: ",
        np.ma.masked_invalid(psnr_diffusion_sr3).mean(),
    )
    print("SR3 (Diffusion) - Average MSE: ", mse_diffusion_sr3.mean())
    print("SR3 (Diffusion) - Average MAE: ", mae_diffusion_sr3.mean())
    print("SR3 (Diffusion) - Average SSIM: ", ssim_diffusion_sr3.mean())

    print(
        "SR3 (Diffusion) - Standard Deviation of PSNR: ",
        np.ma.masked_invalid(psnr_diffusion_sr3).std(),
    )
    print("SR3 (Diffusion) - Standard Deviation of MSE: ", mse_diffusion_sr3.std())
    print("SR3 (Diffusion) - Standard Deviation of MAE: ", mae_diffusion_sr3.std())
    print("SR3 (Diffusion) - Standard Deviation of SSIM: ", ssim_diffusion_sr3.std())


if __name__ == "__main__":
    exp_4()
