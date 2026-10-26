import numpy as np
import pandas as pd


FEATURES = [
    "lag_1",
    "lag_2",
    "ma_5",
    "ma_20",
    "daily_return",
    "volatility_10",
    "price_range",
    "month",
    "day_of_week"
]


def add_features(df):
    d = df.copy().sort_values("date").reset_index(drop=True)

    d["daily_return"] = d["close"].pct_change()
    d["price_range"] = d["high"] - d["low"]

    d["lag_1"] = d["close"].shift(1)
    d["lag_2"] = d["close"].shift(2)

    d["ma_5"] = d["close"].rolling(5).mean()
    d["ma_20"] = d["close"].rolling(20).mean()

    d["volatility_10"] = d["daily_return"].rolling(10).std()

    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    d["day_of_week"] = d["date"].dt.dayofweek
    d["weekday"] = d["date"].dt.day_name()

    d["next_close"] = d["close"].shift(-1)

    return d


def iqr_outliers(series):
    s = series.dropna()

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (series < lower) | (series > upper)

    return lower, upper, mask


def outlier_summary(df, column):
    lower, upper, mask = iqr_outliers(df[column])

    return {
        "column": column,
        "lower_bound": lower,
        "upper_bound": upper,
        "count": int(mask.sum()),
        "percent": float(mask.mean() * 100)
    }


def correlation_table(df):
    cols = [
        "open", "high", "low", "close",
        "daily_return", "price_range",
        "ma_5", "ma_20", "volatility_10"
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].corr()
