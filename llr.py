import os
import joblib
import numpy as np
import util

from scipy import ndimage

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from skimage.util.shape import view_as_windows


def generate_features(data_matrix: np.ndarray, label_matrix: np.ndarray):
    """Generates features for residual training in traditional regression techniques.
    The features consist of the original patch, the first order difference,
    and the second order difference.

    Args:
        data_matrix (np.ndarray): Input data matrix of shape (batch_size, 2, 20, 20)
        label_matrix (np.ndarray): Input label matrix of shape (batch_size, 2, 100, 100)

    Returns:
        (tuple, tuple): _description_
    """

    data_matrix_bicubic = util.bicubic_interpolation(data_matrix)

    label_residual = label_matrix - data_matrix_bicubic

    # Create smaller patches (20x20) from the (100x100) array
    n = data_matrix.shape[0]

    # Feature one: Original patch
    X_ua_upsampled = np.zeros((25 * n, 20 * 20))
    X_va_upsampled = np.zeros((25 * n, 20 * 20))

    # Feature two: First-order difference
    X_ua_first_ord = np.zeros((25 * n, 20 * 20))
    X_va_first_ord = np.zeros((25 * n, 20 * 20))

    # Feature three: Second-order difference
    X_ua_second_ord = np.zeros((25 * n, 20 * 20))
    X_va_second_ord = np.zeros((25 * n, 20 * 20))

    # Output: Recostruction residual
    Y_ua = np.zeros((25 * n, 20 * 20))
    Y_va = np.zeros((25 * n, 20 * 20))

    for i in range(n):
        current_ua_subpatch = (
            data_matrix_bicubic[i, 0, :, :]
            .reshape(5, 20, 5, 20)
            .swapaxes(1, 2)
            .reshape(-1, 20, 20)
        )
        current_va_subpatch = (
            data_matrix_bicubic[i, 1, :, :]
            .reshape(5, 20, 5, 20)
            .swapaxes(1, 2)
            .reshape(-1, 20, 20)
        )

        current_ua_first_ord = ndimage.gaussian_filter(
            current_ua_subpatch, sigma=5, order=1, mode="nearest", axes=(1, 2)
        )

        current_va_first_ord = ndimage.gaussian_filter(
            current_va_subpatch, sigma=5, order=1, mode="nearest", axes=(1, 2)
        )

        current_ua_second_ord = ndimage.gaussian_filter(
            current_ua_subpatch, sigma=5, order=2, mode="nearest", axes=(1, 2)
        )
        current_va_second_ord = ndimage.gaussian_filter(
            current_va_subpatch, sigma=5, order=2, mode="nearest", axes=(1, 2)
        )

        X_ua_upsampled[i * 25 : i * 25 + 25, :] = current_ua_subpatch.reshape(
            -1, 20 * 20
        )
        X_va_upsampled[i * 25 : i * 25 + 25, :] = current_va_subpatch.reshape(
            -1, 20 * 20
        )

        X_ua_first_ord[i * 25 : i * 25 + 25, :] = current_ua_first_ord.reshape(
            -1, 20 * 20
        )
        X_va_first_ord[i * 25 : i * 25 + 25, :] = current_va_first_ord.reshape(
            -1, 20 * 20
        )

        X_ua_second_ord[i * 25 : i * 25 + 25, :] = current_ua_second_ord.reshape(
            -1, 20 * 20
        )
        X_va_second_ord[i * 25 : i * 25 + 25, :] = current_va_second_ord.reshape(
            -1, 20 * 20
        )

        Y_ua[i * 25 : i * 25 + 25, :] = (
            label_residual[i, 0, :, :]
            .reshape(5, 20, 5, 20)
            .swapaxes(1, 2)
            .reshape(-1, 20, 20)
            .reshape(-1, 20 * 20)
        )
        Y_va[i * 25 : i * 25 + 25, :] = (
            label_residual[i, 1, :, :]
            .reshape(5, 20, 5, 20)
            .swapaxes(1, 2)
            .reshape(-1, 20, 20)
            .reshape(-1, 20 * 20)
        )

    X_ua = np.hstack((X_ua_upsampled, X_ua_first_ord, X_ua_second_ord))
    X_va = np.hstack((X_va_upsampled, X_va_first_ord, X_va_second_ord))

    return (X_ua, X_va), (Y_ua, Y_va)


def generate_block_features(block_matrix: np.ndarray):
    """Generates feature blocks for testing datasets. The input is an
    array which represents an overlapping blocks of a test sample.

    Args:
        block_matrix (np.ndarray): array of shape (17, 17, 20, 20) where
        the first two dimensions represent block indicies.

    Returns:
        np.ndarray: features matrix as an array of shape (289, 1200)
    """
    X_upsampled = block_matrix.reshape(-1, 20 * 20)

    X_first_ord = ndimage.gaussian_filter(
        block_matrix, sigma=5, order=1, mode="nearest", axes=[2, 3]
    ).reshape(-1, 20 * 20)
    X_second_ord = ndimage.gaussian_filter(
        block_matrix, sigma=5, order=2, mode="nearest", axes=[2, 3]
    ).reshape(-1, 20 * 20)

    X = np.hstack((X_upsampled, X_first_ord, X_second_ord))
    return X


def random_forest_super_resolution(
    X: np.ndarray, Y: np.ndarray, save_path: str, rf_args: dict, name: str = "rfsr.pkl"
):
    """Trains a random forest model for super resolution.

    Args:
        X (np.ndarray): Data matrix of shape (num_examples, num_features).
        Y (np.ndarray): Label matrix of shape (num_examples, num_outputs).
        save_path (str): Path of where the trained model is saved.
        rf_args (dict): Hyperparameters and arguments of the random forest model.
        name (str, optional): Name of the saved model. Defaults to "rfsr.pkl".
    """
    rf = RandomForestRegressor(**rf_args)
    rf.fit(X, Y)

    joblib.dump(rf, os.path.join(save_path, name))


def predict_block(data_matrix: np.ndarray, model):
    """Perform block-wise prediction of a given test example.

    Args:
        data_matrix (np.ndarray): array of shape (17, 17, 20, 20).
        model : prediction model (e.g., RandomForestRegressor).

    Returns:
        np.ndarray: array of predictions of shape (17, 17, 20, 20).
    """
    X = generate_block_features(data_matrix)

    model.verbose = False
    R = model.predict(X)

    Y = X[:, :400] + R
    Y = Y.reshape(17, 17, 20, 20, order="C")
    return Y


def predict(data_matrix, model_path_ua, model_path_va):
    """Performs predictions on a given dataset.

    Args:
        data_matrix (np.ndarray): data matrix of shape (batch_size, 2, 20, 20).
        model_path_ua (str): path to the ua component prediction model pickle file.
        model_path_va (str): path to the va component prediction model pickle file.

    Returns:
        np.ndarray: array of predictions of shape (batch_size, 2, 100, 100).
    """

    model_ua = joblib.load(model_path_ua)
    model_va = joblib.load(model_path_va)

    n = data_matrix.shape[0]
    predictions = np.zeros((n, 2, 100, 100))

    for i in range(n):
        current_example = data_matrix[i, :, :, :]
        current_example = np.expand_dims(current_example, 0)

        current_example_bicubic = util.bicubic_interpolation(data_matrix)

        for channel_idx in [0, 1]:
            current_example_blocks = view_as_windows(
                current_example_bicubic[i, channel_idx, :, :], 20, 5
            )

            if channel_idx == 0:
                current_example_blocks_predictions = predict_block(
                    current_example_blocks, model_ua
                )
            else:
                current_example_blocks_predictions = predict_block(
                    current_example_blocks, model_va
                )

            current_example_reconstruct = util.reconstruct_blocks(
                current_example_blocks_predictions
            )

            predictions[i, channel_idx, :, :] = current_example_reconstruct

    return predictions


if __name__ == "__main__":
    # Train a basic random forest model
    data_matrix, label_matrix = util.create_subsampled_dataset("dataset/train/", 1)

    X, Y = generate_features(data_matrix, label_matrix)

    rf_args = {
        "n_estimators": 100,
        "max_depth": 12,
        "min_samples_split": 200,
        "n_jobs": -1,
        "verbose": 1,
        "oob_score": True,
    }

    # Train a model for ua and va wind component
    random_forest_super_resolution(X[0], Y[0], "models/", rf_args, name="rfsr_ua.pkl")
    # random_forest_super_resolution(X[1], Y[1], "models/", rf_args, name="rfsr_va.pkl")
    """
    #model_path_ua = "models/rfsr_ua.pkl"
    #model_path_va = "models/rfsr_ua.pkl"

    #Y = predict(data_matrix, model_path_ua, model_path_va)
    """
