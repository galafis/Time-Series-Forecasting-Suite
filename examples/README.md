# Example Datasets

This directory contains sample datasets for testing and demonstrating the Time Series Forecasting Suite.

## Available Datasets

### 1. Sales Data (`sales_data.csv`)
- **Description**: Daily sales data with seasonal patterns and weekly cycles
- **Date Range**: 2022-01-01 to 2024-12-31
- **Features**: 
  - `date`: Date of the observation
  - `sales`: Daily sales amount
- **Characteristics**: Contains yearly seasonality, weekly patterns, and an upward trend
- **Use Case**: Retail sales forecasting, demand prediction

### 2. Temperature Data (`temperature_data.csv`)
- **Description**: Daily temperature measurements with seasonal variation
- **Date Range**: 2020-01-01 to 2024-12-31
- **Features**:
  - `date`: Date of the observation
  - `temperature`: Temperature in Celsius
- **Characteristics**: Strong yearly seasonality, slight warming trend
- **Use Case**: Weather forecasting, climate analysis

### 3. Stock Price Data (`stock_price_data.csv`)
- **Description**: Daily stock prices with random walk characteristics
- **Date Range**: 2021-01-01 to 2024-12-31
- **Features**:
  - `date`: Date of the observation
  - `price`: Stock price
- **Characteristics**: Random walk with drift, realistic volatility
- **Use Case**: Financial forecasting, market analysis

## How to Use

1. **Via Streamlit UI**: Upload any of these CSV files using the file uploader in the sidebar
2. **For Testing**: Use these datasets to validate model performance and compare different algorithms
3. **Custom Data Format**: Use these files as templates for your own time series data

## Data Format Requirements

All CSV files should have:
- A date column (can have any name, you'll select it in the UI)
- A value column (can have any name, you'll select it in the UI)
- Dates in a format parseable by pandas (e.g., 'YYYY-MM-DD')
- No missing dates (or the system will interpolate them)

## Example Usage

```python
import pandas as pd
from forecasting_suite import TimeSeriesForecaster

# Load example data
data = pd.read_csv('examples/sales_data.csv')

# Initialize forecaster
forecaster = TimeSeriesForecaster()
forecaster.load_data(data, date_column='date', value_column='sales')

# Train models
forecaster.train_models()

# Generate forecast
forecast = forecaster.forecast(days=30, model_name='Random Forest')
print(forecast)
```
