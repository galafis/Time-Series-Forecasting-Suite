# Developer Guide

## Quick Start for Contributors

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/galafis/Time-Series-Forecasting-Suite.git
cd Time-Series-Forecasting-Suite

# Install dependencies
pip install -r requirements.txt

# Run tests
python3 -m pytest test_forecasting_suite.py -v

# Run the application
streamlit run forecasting_suite.py
```

### Code Style

This project follows PEP8 guidelines:
- Maximum line length: 120 characters
- Use 4 spaces for indentation
- Follow naming conventions (snake_case for functions/variables, PascalCase for classes)

To check code style:
```bash
flake8 forecasting_suite.py test_forecasting_suite.py --max-line-length=120
```

To auto-format code:
```bash
autopep8 --in-place --aggressive --aggressive forecasting_suite.py
```

### Running Tests

```bash
# Run all tests
python3 -m pytest test_forecasting_suite.py -v

# Run specific test
python3 -m pytest test_forecasting_suite.py::test_load_data -v

# Run with coverage
python3 -m pytest test_forecasting_suite.py --cov=forecasting_suite --cov-report=html
```

### Project Structure

```
Time-Series-Forecasting-Suite/
├── forecasting_suite.py      # Main application code
├── test_forecasting_suite.py # Test suite
├── requirements.txt           # Python dependencies
├── README.md                  # User documentation
├── CHANGELOG.md              # Version history
├── LICENSE                   # MIT License
└── examples/                 # Example datasets
    ├── README.md
    ├── sales_data.csv
    ├── temperature_data.csv
    └── stock_price_data.csv
```

### Key Components

#### TimeSeriesForecaster Class
Main class that handles all forecasting operations:

```python
class TimeSeriesForecaster:
    def __init__(self)
    def load_data(data=None, date_column='date', value_column='value')
    def create_features()
    def analyze_components()
    def train_models(test_size=0.2)
    def forecast(days=30, model_name='Random Forest')
    def create_forecast_plot(model_name='Random Forest', days_history=90)
    def create_components_plot()
```

#### Supported Models
1. **Linear Regression** - Simple baseline model
2. **Random Forest** - Ensemble learning for complex patterns
3. **ARIMA** - Statistical model for time series
4. **Exponential Smoothing** - Weighted average forecasting

### Adding New Features

#### Adding a New Model

1. Add model training in `train_models()`:
```python
# In train_models() method
new_model = YourModel()
new_model.fit(X_train, y_train)
self.models['Your Model'] = new_model
```

2. Add forecasting logic in `forecast()`:
```python
# In forecast() method
elif model_name == 'Your Model':
    forecast_result = model.predict(X_future)
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'forecast': forecast_result
    })
```

3. Add tests in `test_forecasting_suite.py`:
```python
def test_your_model():
    forecaster = TimeSeriesForecaster()
    forecaster.load_data()
    forecaster.train_models()
    forecast = forecaster.forecast(days=30, model_name='Your Model')
    assert not forecast.empty
```

#### Adding UI Components

The UI is built with Streamlit. To add new components:

```python
# In main() function
st.sidebar.header("New Feature")
new_parameter = st.sidebar.slider("Parameter", 1, 100, 50)

# In main content area
st.header("New Visualization")
fig = create_your_plot()
st.plotly_chart(fig, use_container_width=True)
```

### Testing Guidelines

1. **Write tests first** (TDD approach recommended)
2. **Test edge cases**: empty data, single point, missing values
3. **Test all models**: ensure new features work with all forecasting models
4. **Use fixtures** for common setup:
```python
@pytest.fixture
def forecaster_with_data():
    forecaster = TimeSeriesForecaster()
    forecaster.load_data()
    return forecaster
```

### Documentation Standards

- Use docstrings for all functions and classes
- Include parameter types and return types
- Provide usage examples for complex functions
- Keep README.md updated with new features
- Update CHANGELOG.md for each version

### Common Development Tasks

#### Adding Example Dataset
1. Create CSV file with date and value columns
2. Add to `examples/` directory
3. Document in `examples/README.md`
4. Update `.gitignore` if needed

#### Debugging Streamlit App
```bash
# Run with verbose logging
streamlit run forecasting_suite.py --logger.level=debug

# Check console for errors
# Use st.write() for debugging output
```

#### Performance Profiling
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Contribution Workflow

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Make changes and test: `pytest test_forecasting_suite.py -v`
4. Check code style: `flake8 forecasting_suite.py`
5. Commit changes: `git commit -am 'Add your feature'`
6. Push to branch: `git push origin feature/your-feature`
7. Create Pull Request

### Release Checklist

Before releasing a new version:
- [ ] All tests passing
- [ ] Code style checked (flake8)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in appropriate files
- [ ] Screenshots updated if UI changed
- [ ] Example datasets verified

### Getting Help

- Check existing tests for examples
- Review code comments and docstrings
- Refer to library documentation:
  - [Streamlit](https://docs.streamlit.io/)
  - [Plotly](https://plotly.com/python/)
  - [Statsmodels](https://www.statsmodels.org/)
  - [Scikit-learn](https://scikit-learn.org/)

### Best Practices

1. **Keep functions focused** - Each function should do one thing well
2. **Use type hints** - Makes code more maintainable
3. **Handle errors gracefully** - Use try/except for model training
4. **Write self-documenting code** - Clear variable names, logical structure
5. **Test thoroughly** - Aim for high test coverage
6. **Document as you go** - Don't leave it for later

## Questions?

Open an issue on GitHub or check the main README.md for contact information.
