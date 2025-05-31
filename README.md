# Technical Documentation: Stock Price Prediction Analysis

## Overview
This document provides technical documentation for the stock price prediction analysis system implemented in the Jupyter notebook `stock-price-prediction.ipynb`. The system analyzes historical stock price data for major tech companies and builds predictive models using XGBoost to forecast future stock prices.

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Internet connection for downloading financial data

### Environment Setup
1. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv jupyter
2. Turn on the virtual env
   ```
   source jupyter/bin/activate
   ```

2. **Install required packages**:
   ```bash
   pip install yfinance==0.2.61 pandas==2.2.3 numpy==2.2.6 matplotlib==3.10.3 \
               scipy==1.15.3 seaborn==0.13.2 xgboost==3.0.2 scikit-learn==1.6.1 jupyter
   ```


## Data Collection
The system collects historical stock price data using the Yahoo Finance API (yfinance) for the following tech companies:
- Apple (AAPL)
- Microsoft (MSFT)
- Google (GOOG)
- Netflix (NFLX)
- Amazon (AMZN)
- Tesla (TSLA)

Data is collected from January 1, 2015, to January 1, 2025 (or the most recent available date) with daily intervals.

### Data Collection Process
```python
def fetch_data(ticker):
    df = yf.download(ticker, interval='1d', start=start_date, end=end_date)
    df = df[['Close']].copy()
    df.rename(columns={'Close': f'{ticker}_Close'}, inplace=True)
    df.to_csv("data_output.txt", index=False)
    return df
```

## Data Preprocessing

### 1. Initial Data Cleaning
- The closing prices for each stock are extracted and joined into a single DataFrame
- Missing values are removed using `dropna()`

### 2. Statistical Analysis
The system performs basic statistical analysis on the data to understand its distribution:

#### Normality Testing
- Uses Shapiro-Wilk test to check if the data follows a normal distribution
- Calculates and reports the test statistic and p-value for each stock

#### Descriptive Statistics
- Calculates skewness (measure of asymmetry)
- Calculates kurtosis (measure of "tailedness")

### 3. Feature Engineering
The system creates lag features for each stock to use in the predictive models:

```python
def create_lag_features(df, ticker, lags=3):
    for i in range(1, lags + 1):
        df[f'{ticker}_lag_{i}'] = df[f'{ticker}_Close'].shift(i)
    return df
```

For each stock, three lag features are created (previous 3 days' closing prices).

## Model Development

### Model Selection
The system uses XGBoost Regressor for prediction tasks. XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.

### Training Process
For each stock:
1. Features (X) and target (y) are defined:
   - Features: The 3 lag features of the stock
   - Target: The closing price of the stock
2. Data is split into training and testing sets (70/30 split) without shuffling to preserve time-series nature
3. An XGBoost regressor model is trained on the training data
4. The model is used to make predictions on the test data

```python
for ticker in tickers:
    target = f'{ticker}_Close'
    features = [f'{ticker}_lag_{i}' for i in range(1, 4)]
    
    X = data[features]
    y = data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)
    
    model = XGBRegressor()
    model.fit(X_train, y_train)
```

## Model Evaluation

### Performance Metrics
The system evaluates model performance using:
1. Mean Absolute Error (MAE): Average of absolute differences between predicted and actual values
2. Root Mean Square Error (RMSE): Square root of the average of squared differences between predicted and actual values

### Visualization
The results are visualized by:
- Plotting actual vs. predicted stock prices over time
- Using clear labeling and grid for better readability

```python
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.title(f'{ticker} Daily Close: Actual vs Predicted')
plt.legend()
plt.grid(True)
```

## Dependencies
The system requires the following Python libraries:
- yfinance: For fetching stock data
- pandas: For data manipulation
- numpy: For numerical computations
- matplotlib: For data visualization
- scipy: For statistical functions
- seaborn: For enhanced visualizations
- xgboost: For the XGBoost machine learning model
- scikit-learn: For model evaluation metrics and train-test splitting
- yfinance==0.2.61: For fetching stock data
- pandas==2.2.3: For data manipulation
- numpy==2.2.6: For numerical computations
- matplotlib==3.10.3: For data visualization
- scipy==1.15.3: For statistical functions
- seaborn==0.13.2: For enhanced visualizations
- xgboost==3.0.2: For the XGBoost machine learning model
- scikit-learn==1.6.1: For model evaluation metrics and train-test splitting
- jupyter: For running the notebook environment

## Limitations
- The model only uses lagged prices of the same stock for prediction
- The model does not incorporate external factors (economic indicators, news sentiment, etc.)
- Time series cross-validation is not implemented, which could provide a more robust evaluation

### Running the Project
1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   or
   ```bash
   jupyter lab
   ```

2. **Open and Run the Notebook**:
   - Navigate to and open `stock-price-prediction.ipynb`
   - Run cells sequentially using Shift+Enter or use "Kernel > Restart & Run All"

### Customizing the Analysis
- **Changing Stocks**: Modify the `tickers` list at the beginning of the notebook
  ```python
  tickers = ['AAPL', 'MSFT', 'GOOG', 'NFLX', 'AMZN', 'TSLA']
  ```

- **Adjusting Date Range**: Modify the `start_date` and `end_date` variables
  ```python
  start_date = '2015-01-01'
  end_date = '2023-01-01'  
  ```

- **Modifying Model Parameters**: Adjust XGBRegressor parameters for better performance
  ```python
  model = XGBRegressor(
      n_estimators=100,  
      learning_rate=0.1, 
      max_depth=5        
  ```

### Common Issues and Troubleshooting
- **Data Retrieval Issues**: If yfinance fails to download data, check your internet connection or try again later as Yahoo Finance may limit requests
- **Memory Errors**: For large date ranges, ensure your system has sufficient RAM or reduce the date range
- **Poor Model Performance**: Try adding more features, tuning hyperparameters, or using a different model architecture
