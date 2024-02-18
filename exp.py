import numpy as np
import util
import llr
import metrics


def exp_1():
    """Experiment 1: Compares the performance of a ridge regression model
    to bicubic interpolation on the validation.
    """
    metrics_array_rr = metrics.compute_metrics_llr(
        "dataset/val/", "models/lr_ua.pkl", "models/lr_va.pkl", "results/rr_val_metrics"
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


if __name__ == "__main__":
    exp_1()
