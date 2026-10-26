import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from eda import FEATURES


def direction_accuracy(actual, predicted, current_close):
    actual_direction = np.sign(actual - current_close)
    predicted_direction = np.sign(predicted - current_close)
    return (actual_direction == predicted_direction).mean() * 100


def evaluate_model(name, y_true, y_pred, current_close):
    return {
        "Model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "Direction Accuracy %": direction_accuracy(
            y_true.values,
            y_pred,
            current_close.values
        )
    }


def run_models(df):
    data = df.dropna(subset=FEATURES + ["next_close"]).copy()

    # Time-based 80/20 split
    split_index = int(len(data) * 0.80)

    train = data.iloc[:split_index]
    test = data.iloc[split_index:]

    X_train = train[FEATURES]
    y_train = train["next_close"]

    X_test = test[FEATURES]
    y_test = test["next_close"]

    results = []
    predictions = pd.DataFrame({
        "date": test["date"].values,
        "close": test["close"].values,
        "actual": y_test.values
    })

    # Naive baseline
    naive_pred = test["close"].values
    predictions["Naive"] = naive_pred
    results.append(
        evaluate_model("Naive", y_test, naive_pred, test["close"])
    )

    # Linear Regression
    linear = LinearRegression()
    linear.fit(X_train, y_train)
    linear_pred = linear.predict(X_test)
    predictions["Linear Regression"] = linear_pred
    results.append(
        evaluate_model("Linear Regression", y_test, linear_pred, test["close"])
    )

    # Random Forest
    forest = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )
    forest.fit(X_train, y_train)
    forest_pred = forest.predict(X_test)
    predictions["Random Forest"] = forest_pred
    results.append(
        evaluate_model("Random Forest", y_test, forest_pred, test["close"])
    )

    metrics = pd.DataFrame(results)

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": forest.feature_importances_
    }).sort_values("Importance", ascending=False)

    latest = df.dropna(subset=FEATURES).iloc[-1:]
    last_close = float(latest["close"].iloc[0])

    linear.fit(data[FEATURES], data["next_close"])
    forest.fit(data[FEATURES], data["next_close"])

    forecast = pd.DataFrame({
        "Model": ["Naive", "Linear Regression", "Random Forest"],
        "Forecast": [
            last_close,
            float(linear.predict(latest[FEATURES])[0]),
            float(forest.predict(latest[FEATURES])[0])
        ]
    })

    return {
        "metrics": metrics,
        "predictions": predictions,
        "importance": importance,
        "forecast": forecast,
        "train_rows": len(train),
        "test_rows": len(test)
    }
