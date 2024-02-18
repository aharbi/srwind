import util
import llr
import metrics


def exp_1():
    """Experiment 1: Compares the performance of a ridge regression model
    to bicubic interpolation on the validation.
    """
    dataset = "val"

    data_matrix, label_matrix = util.create_subsampled_dataset(
        "dataset/{}/".format(dataset)
    )
 
    model_path_ua = "models/lr_ua.pkl"
    model_path_va = "models/lr_va.pkl"
    prediction_lr = llr.predict(
        data_matrix, model_path_ua, model_path_va
    )

    prediction_bi = util.bicubic_interpolation(data_matrix)

    calc_metrics_bi = metrics.compute_metrics(label_matrix, prediction_bi)
    calc_metrics_lr = metrics.compute_metrics(label_matrix, prediction_lr)

    print(calc_metrics_bi)
    print(calc_metrics_lr)


if __name__ == "__main__":
    exp_1()
