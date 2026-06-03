import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from .eda import EDA
from .metrics import score
from .targets import Targets


TARGET_COLUMNS = ["efs", "efs_time", "cox_hazard", "km_survival", "na_hazard", "event_time"]
DROP_COLUMNS = ["ID", *TARGET_COLUMNS]
ONE_HOT_MODEL_PREFIXES = ("RandomForest", "AdaBoost", "SVR", "linearRegression")


class MD:
    def __init__(self, color, data, cat_cols, early_stop, penalizer, n_splits, random_state=42):
        self.eda = EDA(color, data)
        self.targets = Targets(data, cat_cols, penalizer, n_splits, random_state=random_state)
        self.data = data
        self.cat_cols = cat_cols
        self._early_stop = early_stop

    def create_targets(self):
        self.data = self.targets.cox_hazard()
        self.data = self.targets.km_survival()
        self.data = self.targets.na_hazard()
        self.data = self.targets.event_time()
        return self.data

    def _prepare_features(self, data, title, feature_columns=None):
        data = data.copy()
        for column in self.cat_cols:
            if column in data.columns:
                data[column] = data[column].astype("category")

        if title.startswith(ONE_HOT_MODEL_PREFIXES):
            features = pd.get_dummies(data, columns=[col for col in self.cat_cols if col in data.columns], drop_first=True)
            features = features.drop([col for col in DROP_COLUMNS if col in features.columns], axis=1)
            if feature_columns is not None:
                features = features.reindex(columns=feature_columns, fill_value=0)
            return features

        features = data.drop([col for col in DROP_COLUMNS if col in data.columns], axis=1)
        if feature_columns is not None:
            features = features.reindex(columns=feature_columns, fill_value=0)
        return features

    def train_model(self, params, target, title):
        x = self._prepare_features(self.data, title)
        y = self.data[target]

        models, fold_scores = [], []
        cv, oof_preds = self.targets.prepare_cv()

        for train_index, valid_index in cv.split(x, y):
            x_train = x.iloc[train_index]
            x_valid = x.iloc[valid_index]
            y_train = y.iloc[train_index]
            y_valid = y.iloc[valid_index]

            if title.startswith("LightGBM"):
                try:
                    import lightgbm as lgb
                except ModuleNotFoundError as exc:
                    raise ModuleNotFoundError("Install lightgbm before training LightGBM models") from exc

                model = lgb.LGBMRegressor(**params)
                model.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_valid, y_valid)],
                    eval_metric="rmse",
                    callbacks=[lgb.early_stopping(self._early_stop, verbose=0), lgb.log_evaluation(0)],
                )
            elif title.startswith("CatBoost"):
                try:
                    from catboost import CatBoostRegressor
                except ModuleNotFoundError as exc:
                    raise ModuleNotFoundError("Install catboost before training CatBoost models") from exc

                cat_features = [col for col in self.cat_cols if col in x_train.columns]
                model = CatBoostRegressor(**params, verbose=0, cat_features=cat_features)
                model.fit(
                    x_train,
                    y_train,
                    eval_set=(x_valid, y_valid),
                    early_stopping_rounds=self._early_stop,
                    verbose=0,
                )
            elif title.startswith("RandomForest"):
                model = RandomForestRegressor(**params)
                model.fit(x_train, y_train)
            elif title.startswith("AdaBoost"):
                model = AdaBoostRegressor(**params)
                model.fit(x_train, y_train)
            elif title.startswith("SVR"):
                model = SVR(**params)
                model.fit(x_train, y_train)
            elif title.startswith("linearRegression"):
                model = LinearRegression(**params)
                model.fit(x_train, y_train)
            else:
                raise ValueError(f"Unsupported model type: {title}")

            model.feature_columns_ = list(x.columns)
            models.append(model)
            oof_preds[valid_index] = model.predict(x_valid)

            y_true_fold = self.data.iloc[valid_index][["ID", "efs", "efs_time", "race_group"]].copy()
            y_pred_fold = self.data.iloc[valid_index][["ID"]].copy()
            y_pred_fold["prediction"] = oof_preds[valid_index]
            fold_scores.append(score(y_true_fold, y_pred_fold, "ID"))

        self.eda.plot_cv(fold_scores, title)
        self.targets.validate_model(oof_preds, title)
        return models, oof_preds

    def infer_model(self, data, models, title):
        if not models:
            raise ValueError("At least one model is required for inference")

        feature_columns = getattr(models[0], "feature_columns_", None)
        x = self._prepare_features(data, title, feature_columns=feature_columns)
        return np.mean([model.predict(x) for model in models], axis=0)
