import os
import joblib
import numpy as np
import util

from sklearn.ensemble import RandomForestRegressor


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


if __name__ == "__main__":
    # Train a basic random forest model
    data_matrix, label_matrix = util.create_subsampled_dataset("dataset/train/", 1)

    X = data_matrix[:, 0, :, :].reshape(-1, 20 * 20)
    Y = label_matrix[:, 0, :, :].reshape(-1, 100 * 100)

    print(X.shape, Y.shape)

    rf_args = {
        "n_estimators": 100,
        "max_depth": 12,
        "min_samples_split": 200,
        "n_jobs": -1,
        "verbose": 1,
        "oob_score": True,
    }

    random_forest_super_resolution(X, Y, "models/", rf_args)
