# Gold Price Forecasting & Market Trends

An interactive data analytics and machine learning project focused on exploring historical **XAU/USD gold prices** and predicting the next closing price using simple forecasting models.

The project covers the complete data analysis workflow:

**Raw Data → Data Cleaning → Feature Engineering → Exploratory Data Analysis → Outlier Analysis → Machine Learning → Forecasting**

## Live Dashboard

**Streamlit App:**  
[https://gold-price-forecasting-1.streamlit.app/]

## Project Objective

The main research question is:

> **Can historical gold price patterns help predict the next closing price?**

The project also explores gold price trends, volatility, relationships between market variables, calendar patterns, and unusual market movements.

## Dataset

**Dataset:** XAU/USD Gold Price Historical Data  
**Source:** Kaggle  

The original dataset contains historical gold market information including:

- Date
- Open
- High
- Low
- Close
- Volume

## Data Preparation

Before analysis, the dataset was checked for:

- Missing values
- Duplicate rows
- Invalid price records
- Date formatting
- Data types


## Feature Engineering

Additional variables were created to support the exploratory analysis and machine learning models.

| Feature | Description |
|---|---|
| `daily_return` | Daily percentage change in closing price |
| `price_range` | Difference between daily High and Low |
| `lag_1` | Previous day's closing price |
| `lag_2` | Closing price two days earlier |
| `ma_5` | 5-day moving average |
| `ma_20` | 20-day moving average |
| `volatility_10` | 10-day rolling volatility |
| `month` | Month extracted from the date |
| `day_of_week` | Day of the week |
| `next_close` | Next trading day's closing price and prediction target |

## Exploratory Data Analysis

The EDA includes several types of analysis:

### Univariate Analysis

- Daily return distribution
- Boxplots
- Outlier detection

### Time-Based Analysis

- Historical gold price trend
- Moving averages
- Average closing price by year
- Rolling volatility

### Numerical Relationships

- Open price vs Close price
- Correlation heatmap

### Categorical Relationships

- Average return by month
- Average return by weekday

## Outlier Analysis

The **Interquartile Range (IQR)** method was used to identify unusual observations across important variables such as:

- Open
- High
- Low
- Close
- Daily Return
- Price Range
- Volatility

Outliers were not automatically removed because extreme observations in financial markets may represent genuine market events rather than data errors.

## Machine Learning

Three models were compared for next-day gold price prediction:

### Naive Baseline

Assumes:

**Tomorrow's closing price = Today's closing price**

This provides a simple benchmark that machine learning models should outperform.

### Linear Regression

Used to model linear relationships between historical features and the next closing price.

### Random Forest Regressor

Used to capture more complex and non-linear relationships between market features.

## Model Evaluation

The models were evaluated using:

- **MAE - Mean Absolute Error**
- **RMSE - Root Mean Squared Error**
- **R² Score**
- **Direction Accuracy**

A chronological train/test split was used instead of random shuffling because the dataset is a time series.

## Dashboard

The interactive Streamlit dashboard allows users to:

- Filter the analysis by year
- Explore raw and processed data
- Review feature engineering
- Explore gold price trends
- Analyze volatility
- Compare numerical and categorical relationships
- Select different variables for IQR outlier analysis
- Compare machine learning models
- View feature importance
- View the next-day gold price forecast

## Technologies Used

- Python
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Streamlit
- KaggleHub

## Project Structure

```text
gold-price-forecasting/
│
├── App.py
├── Models.py
├── eda.py
├── data_loader.py
├── Run_All.py
├── requirements.txt
├── README.md
```

## Run the Project Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit dashboard:

```bash
streamlit run App.py
```

## Key Takeaway

Gold prices show clear long-term trends while short-term movements remain difficult to predict.

The project compares simple machine learning approaches against a strong Naive baseline to evaluate whether historical price information provides meaningful predictive value.

## Limitations

The forecasting models mainly use historical gold price information.

Future versions of the project could include external variables such as:

- Interest rates
- Inflation
- USD index
- Oil prices
- Economic indicators
- Geopolitical events

These variables may improve the understanding and forecasting of gold price movements.

## Author

**Layan**

Individual Data Exploration & Dashboard Development Project
