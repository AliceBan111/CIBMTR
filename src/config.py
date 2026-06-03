from pathlib import Path


class CFG:
    project_root = Path(__file__).resolve().parents[1]
    train_path = project_root / "train.csv"
    holdout_prediction_path = project_root / "holdout_test_predictions.csv"

    color = "#31B404"

    batch_size = 32768
    early_stop = 300
    penalizer = 0.01
    n_splits = 7
    random_state = 42
    validation_size = 0.15
    test_size = 0.15
    split_stratify_column = "race_group"

    weights = [1.0, 1.0, 8.0, 4.0, 8.0, 4.0, 6.0, 6.0]

    rf_params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "random_state": 42,
    }

    ada_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "loss": "linear",
    }

    svr_params = {
        "kernel": "rbf",
        "C": 1.0,
        "epsilon": 0.1,
        "gamma": "scale",
    }

    ctb_params = {
        "loss_function": "RMSE",
        "learning_rate": 0.03,
        "random_state": 42,
        "task_type": "CPU",
        "num_trees": 6000,
        "subsample": 0.85,
        "reg_lambda": 8.0,
        "depth": 8,
    }

    lgb_params = {
        "objective": "regression",
        "min_child_samples": 32,
        "num_iterations": 6000,
        "learning_rate": 0.03,
        "extra_trees": True,
        "reg_lambda": 8.0,
        "reg_alpha": 0.1,
        "num_leaves": 64,
        "metric": "rmse",
        "max_depth": 8,
        "device": "cpu",
        "max_bin": 128,
        "verbose": -1,
        "seed": 42,
    }

    cox1_params = {
        "grow_policy": "Depthwise",
        "min_child_samples": 8,
        "loss_function": "Cox",
        "learning_rate": 0.03,
        "random_state": 42,
        "task_type": "CPU",
        "num_trees": 6000,
        "subsample": 0.85,
        "reg_lambda": 8.0,
        "depth": 8,
    }

    cox2_params = {
        "grow_policy": "Lossguide",
        "loss_function": "Cox",
        "learning_rate": 0.03,
        "random_state": 42,
        "task_type": "CPU",
        "num_trees": 6000,
        "subsample": 0.85,
        "reg_lambda": 8.0,
        "num_leaves": 32,
        "depth": 8,
    }

    linear_params = {
        "fit_intercept": True,
        "copy_X": True,
    }
