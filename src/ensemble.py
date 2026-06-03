import numpy as np
import pandas as pd

from .metrics import score


def rank_array(values):
    return pd.Series(values).rank(method="average").to_numpy()


def rank_weighted_average(predictions, weights=None):
    if not predictions:
        raise ValueError("At least one prediction array is required")

    ranked_predictions = np.array([rank_array(prediction) for prediction in predictions])
    if weights is None:
        weights = np.ones(len(predictions), dtype=float)
    weights = np.asarray(weights, dtype=float)

    if len(weights) != len(predictions):
        raise ValueError("The number of weights must match the number of predictions")

    normalized_weights = weights / weights.sum()
    return np.dot(normalized_weights, ranked_predictions)


def evaluate_predictions(dataframe, predictions, title):
    y_true = dataframe[["ID", "efs", "efs_time", "race_group"]].copy()
    y_pred = dataframe[["ID"]].copy()
    y_pred["prediction"] = predictions
    c_index_score = score(y_true, y_pred, "ID")
    print(f"Stratified C-Index Score for {title}: {c_index_score:.4f}")
    return c_index_score


def create_prediction_file(dataframe, predictions, path):
    prediction_file = dataframe[["ID"]].copy()
    prediction_file["prediction"] = predictions
    prediction_file.to_csv(path, index=False)
    return prediction_file
