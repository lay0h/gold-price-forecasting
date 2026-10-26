from data_loader import get_clean
from eda import add_features, outlier_summary
from Models import run_models


if __name__ == "__main__":
    df, report = get_clean(force=True)
    df = add_features(df)

    print("Shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)

    print("\nSummary statistics:")
    print(df.describe().T)

    print("\nMissing values in raw data:")
    print(report["missing_before"])

    print("\nDuplicate rows in raw data:")
    print(report["duplicates_before"])

    print("\nIQR outlier summary:")
    for col in ["open", "high", "low", "close", "daily_return", "price_range", "volatility_10"]:
        print(col, outlier_summary(df, col))

    ml = run_models(df)

    print("\nModel metrics:")
    print(ml["metrics"].round(3))

    print("\nNext-day forecast:")
    print(ml["forecast"].round(2))
