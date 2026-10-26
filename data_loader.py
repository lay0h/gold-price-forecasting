import glob
import os
import pandas as pd

DATASET = "novandraanugrah/xauusd-gold-price-historical-data-2004-2024"
FILE_NAME = "XAU_1d_data.csv"
CLEAN_PATH = "data/clean_gold_daily.csv"


def download_data():
    import kagglehub

    folder = kagglehub.dataset_download(DATASET)
    files = glob.glob(os.path.join(folder, "**", FILE_NAME), recursive=True)

    if not files:
        raise FileNotFoundError(f"{FILE_NAME} was not found.")

    return pd.read_csv(files[0], sep=None, engine="python")


def clean_data(df):
    df = df.copy()

    # 1) Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # 2) Convert date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 3) Convert numeric columns
    numeric_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Data quality BEFORE cleaning
    report = {
        "raw_rows": len(df),
        "missing_before": df.isnull().sum().to_dict(),
        "duplicates_before": int(df.duplicated().sum())
    }

    # 4) Remove rows with missing essential values
    essential = [c for c in ["date", "open", "high", "low", "close"] if c in df.columns]
    df = df.dropna(subset=essential)

    # 5) Remove duplicates
    df = df.drop_duplicates()

    # 6) Remove impossible price rows
    valid_prices = (
        (df["open"] > 0) &
        (df["high"] > 0) &
        (df["low"] > 0) &
        (df["close"] > 0) &
        (df["high"] >= df["low"])
    )
    df = df[valid_prices]

    # 7) Keep 2020 onward
    df = df[df["date"].dt.year >= 2020]

    # 8) Sort data
    df = df.sort_values("date").reset_index(drop=True)

    report["clean_rows"] = len(df)
    report["missing_after"] = df.isnull().sum().to_dict()
    report["duplicates_after"] = int(df.duplicated().sum())

    os.makedirs("data", exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)

    return df, report


def get_clean(force=False):
    if os.path.exists(CLEAN_PATH) and not force:
        df = pd.read_csv(CLEAN_PATH, parse_dates=["date"])
        report = {
            "raw_rows": len(df),
            "clean_rows": len(df),
            "missing_before": df.isnull().sum().to_dict(),
            "missing_after": df.isnull().sum().to_dict(),
            "duplicates_before": int(df.duplicated().sum()),
            "duplicates_after": int(df.duplicated().sum())
        }
        return df, report

    return clean_data(download_data())
