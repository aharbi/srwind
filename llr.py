import os
import joblib
import numpy as np
import util

from scipy import ndimage

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from skimage.util.shape import view_as_windows


def generate_features(
    data_matrix: np.ndarray, label_matrix: np.ndarray, window_size: int
):
    """Generates features for residual training in traditional regression techniques.
    The features consist of the original patch, the first order difference, and the
    second order difference. A rolling window of size window_size is applied to each
    image to generate smaller patches. Then, for each smaller segment, the features
    are computed and sorted in a data matrix.

    Args:
        data_matrix (np.ndarray): Input data matrix of shape (batch_size, 2, 20, 20)
        label_matrix (np.ndarray): Input label matrix of shape (batch_size, 2, 100, 100)
        window_size (int): Size of the square rolling window applied to each patch.

    Returns:
        (tuple, tuple): First tuple contains the data matricies for each wind componenet.
        The shape of each matrix is (batch_size * num_patches, 3 * window_size * window_size).
        The second tuple contains the labels matrix for each wind component, which represent the
        residual. The shape of the labels matrix is (batch_size * num_patches, window_size * window_size)
    """

    data_matrix_bicubic = util.bicubic_interpolation(data_matrix)

    label_residual = label_matrix - data_matrix_bicubic

    batch_size = data_matrix_bicubic.shape[0]

    num_rows = int(data_matrix_bicubic.shape[2] / window_size)
    num_cols = int(data_matrix_bicubic.shape[3] / window_size)

    num_patches = num_rows * num_cols

    # Feature one: Original patch
    X_ua_upsampled = np.zeros((num_patches * batch_size, window_size * window_size))
    X_va_upsampled = np.zeros((num_patches * batch_size, window_size * window_size))

    # Feature two: First-order difference
    X_ua_first_ord = np.zeros((num_patches * batch_size, window_size * window_size))
    X_va_first_ord = np.zeros((num_patches * batch_size, window_size * window_size))

    # Feature three: Second-order difference
    X_ua_second_ord = np.zeros((num_patches * batch_size, window_size * window_size))
    X_va_second_ord = np.zeros((num_patches * batch_size, window_size * window_size))

    # Output: Recostruction residual
    Y_ua = np.zeros((num_patches * batch_size, window_size * window_size))
    Y_va = np.zeros((num_patches * batch_size, window_size * window_size))

    for i in range(batch_size):
        current_ua_subpatch = (
            data_matrix_bicubic[i, 0, :, :]
            .reshape(num_rows, window_size, num_cols, window_size)
            .swapaxes(1, 2)
            .reshape(-1, window_size, window_size)
        )
        current_va_subpatch = (
            data_matrix_bicubic[i, 1, :, :]
            .reshape(num_rows, window_size, num_cols, window_size)
            .swapaxes(1, 2)
            .reshape(-1, window_size, window_size)
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

        X_ua_upsampled[i * num_patches : i * num_patches + num_patches, :] = (
            current_ua_subpatch.reshape(-1, window_size * window_size)
        )
        X_va_upsampled[i * num_patches : i * num_patches + num_patches, :] = (
            current_va_subpatch.reshape(-1, window_size * window_size)
        )

        X_ua_first_ord[i * num_patches : i * num_patches + num_patches, :] = (
            current_ua_first_ord.reshape(-1, window_size * window_size)
        )
        X_va_first_ord[i * num_patches : i * num_patches + num_patches, :] = (
            current_va_first_ord.reshape(-1, window_size * window_size)
        )

        X_ua_second_ord[i * num_patches : i * num_patches + num_patches, :] = (
            current_ua_second_ord.reshape(-1, window_size * window_size)
        )
        X_va_second_ord[i * num_patches : i * num_patches + num_patches, :] = (
            current_va_second_ord.reshape(-1, window_size * window_size)
        )

        Y_ua[i * num_patches : i * num_patches + num_patches, :] = (
            label_residual[i, 0, :, :]
            .reshape(num_rows, window_size, num_cols, window_size)
            .swapaxes(1, 2)
            .reshape(-1, window_size, window_size)
            .reshape(-1, window_size * window_size)
        )
        Y_va[i * num_patches : i * num_patches + num_patches, :] = (
            label_residual[i, 1, :, :]
            .reshape(num_rows, window_size, num_cols, window_size)
            .swapaxes(1, 2)
            .reshape(-1, window_size, window_size)
            .reshape(-1, window_size * window_size)
        )

    X_ua = np.hstack((X_ua_upsampled, X_ua_first_ord, X_ua_second_ord))
    X_va = np.hstack((X_va_upsampled, X_va_first_ord, X_va_second_ord))

    return (X_ua, X_va), (Y_ua, Y_va)


def generate_block_features(block_matrix: np.ndarray):
    """Generates feature blocks for testing datasets. The input is an
    array which represents an overlapping blocks of a test sample.

    Args:
        block_matrix (np.ndarray): array of shape (num_rows, num_cols, window_size, window_size).

    Returns:
        np.ndarray: features matrix as an array of shape (num_rows * num_cols, 1200)
    """
    window_size = block_matrix.shape[2]

    X_upsampled = block_matrix.reshape(-1, window_size * window_size)

    X_first_ord = ndimage.gaussian_filter(
        block_matrix, sigma=5, order=1, mode="nearest", axes=[2, 3]
    ).reshape(-1, window_size * window_size)
    X_second_ord = ndimage.gaussian_filter(
        block_matrix, sigma=5, order=2, mode="nearest", axes=[2, 3]
    ).reshape(-1, window_size * window_size)

    X = np.hstack((X_upsampled, X_first_ord, X_second_ord))
    return X


def random_forest_super_resolution(
    X: np.ndarray,
    Y: np.ndarray,
    save_path: str,
    rf_args: dict,
    pca_components: int,
    name: str = "rfsr.pkl",
):
    """Trains a random forest model for super resolution.

    Args:
        X (np.ndarray): Data matrix of shape (num_examples, num_features).
        Y (np.ndarray): Label matrix of shape (num_examples, num_outputs).
        save_path (str): Path of where the trained model is saved.
        rf_args (dict): Hyperparameters and arguments of the random forest model.
        pca_components (int): Number of principle componenets to keep in PCA.
        name (str, optional): Name of the saved model. Defaults to "rfsr.pkl".
    """
    scaler = MinMaxScaler()
    scaler.fit(X)

    X = scaler.transform(X)

    pca = PCA(n_components=pca_components)
    X_d = pca.fit_transform(X)

    rf = RandomForestRegressor(**rf_args)
    rf.fit(X_d, Y)

    joblib.dump(rf, os.path.join(save_path, name))
    joblib.dump(pca, os.path.join(save_path, "pca_" + name))
    joblib.dump(scaler, os.path.join(save_path, "scaler_" + name))


def linear_regression_super_resolution(
    X: np.ndarray,
    Y: np.ndarray,
    save_path: str,
    lr_args: dict,
    pca_components: int,
    name: str = "lr.pkl",
):
    """Trains a linear regression model for super resolution.

    Args:
        X (np.ndarray): Data matrix of shape (num_examples, num_features).
        Y (np.ndarray): Label matrix of shape (num_examples, num_outputs).
        save_path (str): Path of where the trained model is saved.
        lr_args (dict): Hyperparameters and arguments of the random forest model.
        pca_components (int): Number of principle componenets to keep in PCA.
        name (str, optional): Name of the saved model. Defaults to "lr.pkl".
    """
    scaler = MinMaxScaler()
    scaler.fit(X)

    X = scaler.transform(X)

    pca = PCA(n_components=pca_components)
    X_d = pca.fit_transform(X)

    lr = Ridge(**lr_args)
    lr.fit(X_d, Y)

    joblib.dump(lr, os.path.join(save_path, name))
    joblib.dump(pca, os.path.join(save_path, "pca_" + name))
    joblib.dump(scaler, os.path.join(save_path, "scaler_" + name))


def predict_block(data_matrix: np.ndarray, model, pca, scaler):
    """Perform block-wise prediction of a given test example.

    Args:
        data_matrix (np.ndarray): array of shape (num_rows, num_cols, window_size, window_size).
        model : prediction model (e.g., RandomForestRegressor).

    Returns:
        np.ndarray: array of predictions of shape (num_rows, num_cols, window_size, window_size).
    """
    num_rows, num_cols = data_matrix.shape[0], data_matrix.shape[1]
    window_size = data_matrix.shape[2]

    X = generate_block_features(data_matrix)
    X = scaler.transform(X)
    
    X_d = pca.transform(X)

    model.verbose = False
    R = model.predict(X_d)

    Y = X[:, : window_size * window_size] + R
    Y = Y.reshape(num_rows, num_cols, window_size, window_size, order="C")
    return Y


def predict(
    data_matrix,
    model_path_ua,
    pca_path_ua,
    scaler_path_ua,
    model_path_va,
    pca_path_va,
    scaler_path_va,
    window_size,
    stride,
):
    """Performs predictions on a given dataset.

    Args:
        data_matrix (np.ndarray): data matrix of shape (batch_size, 2, 20, 20).
        model_path_ua (str): path to the ua component prediction model pickle file.
        model_path_va (str): path to the va component prediction model pickle file.

    Returns:
        np.ndarray: array of predictions of shape (batch_size, 2, 100, 100).
    """

    model_ua = joblib.load(model_path_ua)
    pca_ua = joblib.load(pca_path_ua)
    scaler_ua = joblib.load(scaler_path_ua)

    model_va = joblib.load(model_path_va)
    pca_va = joblib.load(pca_path_va)
    scaler_va = joblib.load(scaler_path_va)

    n = data_matrix.shape[0]
    predictions = np.zeros((n, 2, 100, 100))

    for i in range(n):
        current_example = data_matrix[i, :, :, :]
        current_example = np.expand_dims(current_example, 0)

        current_example_bicubic = util.bicubic_interpolation(data_matrix)

        for channel_idx in [0, 1]:
            current_example_blocks = view_as_windows(
                current_example_bicubic[i, channel_idx, :, :], window_size, stride
            )

            if channel_idx == 0:
                current_example_blocks_predictions = predict_block(
                    current_example_blocks, model_ua, pca_ua, scaler_ua
                )
            else:
                current_example_blocks_predictions = predict_block(
                    current_example_blocks, model_va, pca_va, scaler_va
                )

            current_example_reconstruct = util.reconstruct_blocks(
                current_example_blocks_predictions, stride
            )

            predictions[i, channel_idx, :, :] = current_example_reconstruct

    return predictions
