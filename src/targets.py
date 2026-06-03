import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .metrics import score


class Targets:
    def __init__(self, data, cat_cols, penalizer, n_splits, random_state=42):
        self.data = data
        self.cat_cols = cat_cols
        self._length = len(self.data)
        self._penalizer = penalizer
        self._n_splits = n_splits
        self._random_state = random_state

    def prepare_cv(self):
        oof_preds = np.zeros(self._length)
        cv = KFold(n_splits=self._n_splits, shuffle=True, random_state=self._random_state)
        return cv, oof_preds

    def validate_model(self, preds, title):
        y_true = self.data[["ID", "efs", "efs_time", "race_group"]].copy()
        y_pred = self.data[["ID"]].copy()
        y_pred["prediction"] = preds
        c_index_score = score(y_true, y_pred, "ID")
        print(f"Overall Stratified C-Index Score for {title}: {c_index_score:.4f}")
        return c_index_score

    def cox_hazard(self):
        try:
            from lifelines import CoxPHFitter
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Install lifelines before creating Cox targets") from exc

        cv, oof_preds = self.prepare_cv()
        data = pd.get_dummies(self.data, columns=self.cat_cols, drop_first=True).drop("ID", axis=1)

        for train_index, valid_index in cv.split(data):
            train_data = data.iloc[train_index]
            valid_data = data.iloc[valid_index]

            train_data = train_data.loc[:, train_data.nunique() > 1]
            valid_data = valid_data[train_data.columns]

            model = CoxPHFitter(penalizer=self._penalizer)
            model.fit(train_data, duration_col="efs_time", event_col="efs")
            oof_preds[valid_index] = model.predict_partial_hazard(valid_data)

        self.data["cox_hazard"] = oof_preds
        self.validate_model(oof_preds, "Cox")
        return self.data

    def km_survival(self):
        try:
            from lifelines import KaplanMeierFitter
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Install lifelines before creating Kaplan-Meier targets") from exc

        cv, oof_preds = self.prepare_cv()

        for train_index, valid_index in cv.split(self.data):
            train_data = self.data.iloc[train_index]
            valid_data = self.data.iloc[valid_index]

            model = KaplanMeierFitter()
            model.fit(durations=train_data["efs_time"], event_observed=train_data["efs"])
            oof_preds[valid_index] = model.survival_function_at_times(valid_data["efs_time"]).values

        self.data["km_survival"] = oof_preds
        self.validate_model(oof_preds, "Kaplan-Meier")
        return self.data

    def na_hazard(self):
        try:
            from lifelines import NelsonAalenFitter
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Install lifelines before creating Nelson-Aalen targets") from exc

        cv, oof_preds = self.prepare_cv()

        for train_index, valid_index in cv.split(self.data):
            train_data = self.data.iloc[train_index]
            valid_data = self.data.iloc[valid_index]

            model = NelsonAalenFitter()
            model.fit(durations=train_data["efs_time"], event_observed=train_data["efs"])
            oof_preds[valid_index] = -model.cumulative_hazard_at_times(valid_data["efs_time"]).values

        self.data["na_hazard"] = oof_preds
        self.validate_model(oof_preds, "Nelson-Aalen")
        return self.data

    def event_time(self):
        self.data["event_time"] = self.data.efs_time.copy()
        self.data.loc[self.data.efs == 0, "event_time"] *= -1
        return self.data
