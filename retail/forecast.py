"""Expanding-window validation followed by one untouched chronological test."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from retail.data import calendar_series

FEATURES = ["lag_1", "lag_7", "rolling_7", "dayofweek", "month"]
MODELS = ["Yesterday", "Last week", "Random Forest"]


def make_features(sales):
    out = pd.DataFrame({"actual": sales})
    out["lag_1"] = sales.shift(1)
    out["lag_7"] = sales.shift(7)
    out["rolling_7"] = sales.shift(1).rolling(7, min_periods=7).mean()
    out["dayofweek"] = sales.index.dayofweek
    out["month"] = sales.index.month
    # Explicit columns: unrelated missing product metadata cannot delete rows.
    return out.dropna(subset=["actual"] + FEATURES)


def make_splits(start, end):
    dates = pd.date_range(start, end)
    if len(dates) < 100:
        raise ValueError("At least 100 calendar days are required for this evaluation.")
    n_dev = int(len(dates) * 0.8)
    block = max(14, int(n_dev * 0.15))
    first = n_dev - 3 * block
    folds = []
    for i in range(3):
        lo = first + i * block
        folds.append({"fold": i + 1, "train_end": dates[lo - 1],
                      "start": dates[lo], "end": dates[lo + block - 1]})
    return {"folds": folds, "selection_end": folds[0]["train_end"],
            "development_end": dates[n_dev - 1], "test_start": dates[n_dev],
            "test_end": dates[-1]}


def fit_forest(train):
    forest = RandomForestRegressor(n_estimators=200, max_depth=10,
                                   min_samples_leaf=2, random_state=42, n_jobs=1)
    return forest.fit(train[FEATURES], train.actual)


def predict_one_day(frame, forest, model):
    if model == "Yesterday":
        return frame.lag_1.to_numpy()
    if model == "Last week":
        return frame.lag_7.to_numpy()
    if model == "Random Forest":
        return forest.predict(frame[FEATURES])
    raise ValueError(f"Unknown model: {model}")


def score(actual, predicted, training_sales):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    scale = training_sales.diff(7).abs().mean()
    mae = mean_absolute_error(actual, predicted)
    denominator = np.abs(actual).sum()
    return {"mae": float(mae),
            "wape_pct": float(np.abs(actual - predicted).sum() / denominator * 100) if denominator else np.nan,
            "mase": float(mae / scale) if scale > 0 else np.nan,
            "bias": float(np.mean(predicted - actual)), "n_days": len(actual)}


def run_evaluation(daily, top_n=10):
    start, end = daily.date.min(), daily.date.max()
    splits = make_splits(start, end)
    # Select the assortment BEFORE any validation or test observation is seen.
    ranking = daily[daily.date <= splits["selection_end"]].groupby("product_id").sales.sum()
    ids = ranking.sort_values(ascending=False).head(top_n).index.tolist()
    names = daily.groupby("product_id").product.first().to_dict()
    fold_rows, test_rows, prediction_rows, selections, coverage, excluded = [], [], [], [], [], []
    series, fitted, validation_forests = {}, {}, {}
    for pid in ids:
        ts = calendar_series(daily, pid, start, end)
        series[pid] = ts
        features = make_features(ts.sales)
        test = features.loc[splits["test_start"]:splits["test_end"]]
        coverage.append({"product_id": pid, "product": names[pid],
                         "calendar_days": len(ts), "observed_days": int(ts.sales.notna().sum()),
                         "missing_days": int(ts.sales.isna().sum()),
                         "test_calendar_days": len(ts.loc[splits["test_start"]:]),
                         "test_scored_days": len(test)})
        reason = None
        for fold in splits["folds"]:
            if len(features.loc[:fold["train_end"]]) < 60 or len(features.loc[fold["start"]:fold["end"]]) < 7:
                reason = f"Insufficient observations in validation fold {fold['fold']} (minimum 60 training / 7 validation days)."
                break
        if len(test) < 7:
            reason = "Fewer than 7 eligible final test days."
        if reason:
            excluded.append({"product_id": pid, "product": names[pid], "reason": reason})
            continue
        product_folds = []
        for fold in splits["folds"]:
            train = features[features.index <= fold["train_end"]]
            valid = features.loc[fold["start"]:fold["end"]]
            forest = fit_forest(train)
            validation_forests[pid, fold["fold"]] = forest
            for model in MODELS:
                predicted = predict_one_day(valid, forest, model)
                row = {"product_id": pid, "product": names[pid], "model": model,
                       "fold": fold["fold"], "train_end": fold["train_end"],
                       "validation_start": fold["start"], "validation_end": fold["end"],
                       **score(valid.actual, predicted, ts.loc[:fold["train_end"], "sales"])}
                fold_rows.append(row)
                product_folds.append(row)
        # Average absolute error weighted by evaluated days across validation folds.
        table = pd.DataFrame(product_folds)
        validation_mae = table.groupby("model", sort=False).apply(
            lambda part: np.average(part.mae, weights=part.n_days), include_groups=False
        )
        selected = min(MODELS, key=lambda model: validation_mae[model])
        baseline = min(MODELS[:2], key=lambda model: validation_mae[model])
        selections.append({"product_id": pid, "product": names[pid], "selected_model": selected,
                           "baseline_model": baseline, "validation_mae": validation_mae[selected]})
        train = features.loc[:splits["development_end"]]
        fitted[pid] = fit_forest(train)
        for model in MODELS:
            predicted = predict_one_day(test, fitted[pid], model)
            test_rows.append({"product_id": pid, "product": names[pid], "model": model,
                              "selected": model == selected, "validation_baseline": model == baseline,
                              **score(test.actual, predicted, ts.loc[:splits["development_end"], "sales"])})
            prediction_rows.extend({"product_id": pid, "product": names[pid], "date": day,
                                    "model": model, "actual": float(actual), "prediction": float(pred)}
                                   for day, actual, pred in zip(test.index, test.actual, predicted))
    if not selections:
        raise ValueError("No selected products have sufficient chronological coverage.")
    return {"splits": splits, "series": series, "forests": fitted,
            "validation_forests": validation_forests,
            "validation": pd.DataFrame(fold_rows), "test": pd.DataFrame(test_rows),
            "predictions": pd.DataFrame(prediction_rows), "selection": pd.DataFrame(selections),
            "coverage": pd.DataFrame(coverage), "excluded": pd.DataFrame(excluded, columns=["product_id", "product", "reason"])}


def forecast_horizon(sales, origins, horizon, model, forest):
    """Forecast each horizon from ONE origin, without future actual sales.

    For an origin t, only t-7..t-1 enter the initial history. Subsequent
    predictions are fed back recursively, including for the weekly baseline.
    """
    if horizon < 1:
        raise ValueError("Horizon must be positive.")
    origins = pd.DatetimeIndex(origins)
    history = np.array([[sales.get(t - pd.Timedelta(days=k), np.nan)
                         for k in range(7, 0, -1)] for t in origins], dtype=float).reshape(-1, 7)
    valid = np.isfinite(history).all(axis=1)
    output = np.full((len(origins), horizon), np.nan)
    hist = history[valid]
    for step in range(horizon):
        dates = origins[valid] + pd.Timedelta(days=step)
        if not len(dates):
            break
        if model == "Yesterday":
            pred = hist[:, -1]
        elif model == "Last week":
            pred = hist[:, -7]
        elif model == "Random Forest":
            features = pd.DataFrame({"lag_1": hist[:, -1], "lag_7": hist[:, -7],
                                     "rolling_7": hist[:, -7:].mean(axis=1),
                                     "dayofweek": dates.dayofweek, "month": dates.month})
            pred = forest.predict(features[FEATURES])
        else:
            raise ValueError(f"Unknown model: {model}")
        pred = np.maximum(0, pred)
        output[valid, step] = pred
        hist = np.column_stack([hist, pred])[:, -7:]
    return pd.DataFrame(output, index=origins, columns=range(1, horizon + 1))
