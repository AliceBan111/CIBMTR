import numpy as np
import pandas as pd
import pandas.api.types


class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    try:
        from lifelines.utils import concordance_index
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install lifelines before scoring survival predictions") from exc

    solution = solution.copy()
    submission = submission.copy()

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    event_label = "efs"
    interval_label = "efs_time"
    prediction_label = "prediction"

    for column in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[column]):
            raise ParticipantVisibleError(f"Submission column {column} must be numeric")

    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    grouped_indices = dict(merged_df.groupby(["race_group"]).groups)
    metric_list = []

    for race in grouped_indices:
        indices = sorted(grouped_indices[race])
        merged_race_df = merged_df.iloc[indices]
        race_c_index = concordance_index(
            merged_race_df[interval_label],
            -merged_race_df[prediction_label],
            merged_race_df[event_label],
        )
        metric_list.append(race_c_index)

    return float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))
