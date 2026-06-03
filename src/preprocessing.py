import pandas as pd
from sklearn.model_selection import train_test_split


NUMERIC_COLUMNS = [
    "hla_high_res_8",
    "hla_low_res_8",
    "hla_high_res_6",
    "hla_low_res_6",
    "hla_high_res_10",
    "hla_low_res_10",
    "hla_match_dqb1_high",
    "hla_match_dqb1_low",
    "hla_match_drb1_high",
    "hla_match_drb1_low",
    "hla_nmdp_6",
    "year_hct",
    "hla_match_a_high",
    "hla_match_a_low",
    "hla_match_b_high",
    "hla_match_b_low",
    "hla_match_c_high",
    "hla_match_c_low",
    "donor_age",
    "age_at_hct",
    "comorbidity_score",
    "karnofsky_score",
    "efs",
    "efs_time",
]


def cast_datatypes(dataframe):
    dataframe = dataframe.copy()
    for column in dataframe.columns:
        if column in NUMERIC_COLUMNS:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce").fillna(-1).astype("float32")
        else:
            dataframe[column] = dataframe[column].fillna("Unknown").astype("string")

    dataframe["ID"] = dataframe["ID"].astype("int32")
    return dataframe


def load_training_data(path) -> pd.DataFrame:
    return cast_datatypes(pd.read_csv(path))


def get_categorical_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        column
        for column in dataframe.columns
        if pd.api.types.is_object_dtype(dataframe[column])
        or pd.api.types.is_string_dtype(dataframe[column])
        or isinstance(dataframe[column].dtype, pd.CategoricalDtype)
    ]


def info_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": dataframe.columns,
            "dtype": dataframe.dtypes.astype(str).values,
            "missing": dataframe.isna().sum().values,
            "unique": dataframe.nunique(dropna=False).values,
        }
    )


def split_dataset(
    dataframe: pd.DataFrame,
    validation_size: float,
    test_size: float,
    random_state: int,
    stratify_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if validation_size <= 0 or test_size <= 0:
        raise ValueError("Validation and test sizes must be positive")
    if validation_size + test_size >= 1:
        raise ValueError("Validation and test sizes must sum to less than 1")

    stratify = dataframe[stratify_column] if stratify_column in dataframe.columns else None
    holdout_size = validation_size + test_size
    train_data, holdout_data = train_test_split(
        dataframe,
        test_size=holdout_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )

    relative_test_size = test_size / holdout_size
    holdout_stratify = holdout_data[stratify_column] if stratify_column in holdout_data.columns else None
    validation_data, test_data = train_test_split(
        holdout_data,
        test_size=relative_test_size,
        random_state=random_state,
        shuffle=True,
        stratify=holdout_stratify,
    )

    return (
        train_data.reset_index(drop=True),
        validation_data.reset_index(drop=True),
        test_data.reset_index(drop=True),
    )
