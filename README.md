# Time Series Forecasting Suite

[English](#english) | [Português](#português)

## English

### Overview
Comprehensive time series forecasting suite with multiple algorithms, interactive visualizations, and professional web interface. Features ARIMA, exponential smoothing, machine learning models, and advanced analytics for accurate time series prediction and analysis.

### Features
- **Multiple Algorithms**: ARIMA, Exponential Smoothing, Linear Regression, Random Forest
- **Interactive Dashboard**: Streamlit-based web interface with real-time updates
- **Data Visualization**: Professional charts with Plotly and Matplotlib
- **Forecast Accuracy**: Multiple evaluation metrics (MAE, RMSE, MAPE)
- **Seasonal Decomposition**: Trend, seasonal, and residual analysis
- **Data Upload**: Support for CSV files and sample datasets
- **Export Results**: Download forecasts and visualizations
- **Model Comparison**: Side-by-side algorithm performance analysis

### Technologies Used
- **Python 3.8+**
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Plotly**: Interactive visualizations
- **Matplotlib & Seaborn**: Statistical plotting
- **Statsmodels**: Statistical modeling

### Installation

1. Clone the repository:
```bash
git clone https://github.com/galafis/Time-Series-Forecasting-Suite.git
cd Time-Series-Forecasting-Suite
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run forecasting_suite.py
```

4. Open your browser to `http://localhost:8501`

### Usage

#### Web Interface
1. **Data Input**: Upload CSV file or use sample data
2. **Data Exploration**: View time series plots and statistics
3. **Model Selection**: Choose forecasting algorithm and parameters
4. **Generate Forecast**: Create predictions for specified periods
5. **Evaluate Results**: Compare accuracy metrics and visualizations
6. **Export**: Download forecasts and charts

#### Supported Data Formats
- **CSV Files**: Date column + value column(s)
- **Date Formats**: Automatic detection of common formats
- **Frequency**: Daily, weekly, monthly, quarterly, yearly
- **Missing Values**: Automatic handling and interpolation

#### Python API
```python
from forecasting_suite import TimeSeriesForecaster

# Initialize forecaster
forecaster = TimeSeriesForecaster()

# Load data
data = forecaster.load_data('your_data.csv', date_col='date', value_col='value')

# Fit model and forecast
forecaster.fit_arima(data)
forecast = forecaster.forecast(periods=30)

# Evaluate accuracy
metrics = forecaster.evaluate_forecast(actual, predicted)
print(f"MAE: {metrics['mae']:.2f}")
print(f"RMSE: {metrics['rmse']:.2f}")
```

### Forecasting Models

#### ARIMA (AutoRegressive Integrated Moving Average)
- **Best for**: Stationary time series with trends
- **Parameters**: Auto-selection using AIC/BIC criteria
- **Seasonality**: SARIMA for seasonal patterns

#### Exponential Smoothing
- **Simple**: Basic trend forecasting
- **Double**: Trend and level smoothing
- **Triple (Holt-Winters)**: Trend, level, and seasonality

#### Machine Learning Models
- **Linear Regression**: Simple trend-based forecasting
- **Random Forest**: Non-linear pattern recognition
- **Feature Engineering**: Lag features, rolling statistics

#### Ensemble Methods
- **Model Averaging**: Combine multiple forecasts
- **Weighted Ensemble**: Performance-based weighting
- **Stacking**: Meta-learning approach

### Features

#### Data Preprocessing
- **Missing Value Handling**: Forward fill, interpolation, mean imputation
- **Outlier Detection**: Statistical and visual identification
- **Stationarity Testing**: ADF test and differencing
- **Seasonal Decomposition**: STL decomposition

#### Visualization
- **Time Series Plots**: Interactive line charts with zoom
- **Forecast Plots**: Predictions with confidence intervals
- **Residual Analysis**: Error distribution and autocorrelation
- **Seasonal Plots**: Monthly/quarterly patterns

#### Model Evaluation
- **Accuracy Metrics**: MAE, RMSE, MAPE, SMAPE
- **Cross-Validation**: Time series split validation
- **Residual Analysis**: Normality and independence tests
- **Forecast Intervals**: Confidence and prediction intervals

### Sample Datasets
- **Stock Prices**: Daily stock market data
- **Sales Data**: Monthly retail sales
- **Weather**: Temperature and precipitation
- **Economic Indicators**: GDP, inflation, unemployment

### Configuration
Customize forecasting parameters:
```python
config = {
    'arima_order': (1, 1, 1),
    'seasonal_order': (1, 1, 1, 12),
    'forecast_periods': 30,
    'confidence_level': 0.95
}
```

### Performance Tips
- **Data Quality**: Clean and consistent time series data
- **Model Selection**: Use cross-validation for model comparison
- **Seasonality**: Consider seasonal patterns in your data
- **Forecast Horizon**: Shorter forecasts are generally more accurate

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Português

### Visão Geral
Suite abrangente de previsão de séries temporais com múltiplos algoritmos, visualizações interativas e interface web profissional. Apresenta ARIMA, suavização exponencial, modelos de machine learning e análises avançadas para previsão e análise precisas de séries temporais.

### Funcionalidades
- **Múltiplos Algoritmos**: ARIMA, Suavização Exponencial, Regressão Linear, Random Forest
- **Dashboard Interativo**: Interface web baseada em Streamlit com atualizações em tempo real
- **Visualização de Dados**: Gráficos profissionais com Plotly e Matplotlib
- **Precisão de Previsão**: Múltiplas métricas de avaliação (MAE, RMSE, MAPE)
- **Decomposição Sazonal**: Análise de tendência, sazonalidade e resíduos
- **Upload de Dados**: Suporte para arquivos CSV e datasets de exemplo
- **Exportar Resultados**: Download de previsões e visualizações
- **Comparação de Modelos**: Análise de performance lado a lado dos algoritmos

### Tecnologias Utilizadas
- **Python 3.8+**
- **Streamlit**: Framework de aplicação web
- **Pandas**: Manipulação e análise de dados
- **NumPy**: Computação numérica
- **Scikit-learn**: Algoritmos de machine learning
- **Plotly**: Visualizações interativas
- **Matplotlib & Seaborn**: Plotagem estatística
- **Statsmodels**: Modelagem estatística

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/galafis/Time-Series-Forecasting-Suite.git
cd Time-Series-Forecasting-Suite
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute a aplicação:
```bash
streamlit run forecasting_suite.py
```

4. Abra seu navegador em `http://localhost:8501`

### Uso

#### Interface Web
1. **Entrada de Dados**: Upload de arquivo CSV ou use dados de exemplo
2. **Exploração de Dados**: Visualize gráficos de séries temporais e estatísticas
3. **Seleção de Modelo**: Escolha algoritmo de previsão e parâmetros
4. **Gerar Previsão**: Crie predições para períodos especificados
5. **Avaliar Resultados**: Compare métricas de precisão e visualizações
6. **Exportar**: Download de previsões e gráficos

#### Formatos de Dados Suportados
- **Arquivos CSV**: Coluna de data + coluna(s) de valor
- **Formatos de Data**: Detecção automática de formatos comuns
- **Frequência**: Diária, semanal, mensal, trimestral, anual
- **Valores Ausentes**: Tratamento automático e interpolação

#### API Python
```python
from forecasting_suite import TimeSeriesForecaster

# Inicializar forecaster
forecaster = TimeSeriesForecaster()

# Carregar dados
data = forecaster.load_data('seus_dados.csv', date_col='data', value_col='valor')

# Ajustar modelo e prever
forecaster.fit_arima(data)
forecast = forecaster.forecast(periods=30)

# Avaliar precisão
metrics = forecaster.evaluate_forecast(real, previsto)
print(f"MAE: {metrics['mae']:.2f}")
print(f"RMSE: {metrics['rmse']:.2f}")
```

### Modelos de Previsão

#### ARIMA (AutoRegressive Integrated Moving Average)
- **Melhor para**: Séries temporais estacionárias com tendências
- **Parâmetros**: Seleção automática usando critérios AIC/BIC
- **Sazonalidade**: SARIMA para padrões sazonais

#### Suavização Exponencial
- **Simples**: Previsão básica de tendência
- **Dupla**: Suavização de tendência e nível
- **Tripla (Holt-Winters)**: Tendência, nível e sazonalidade

#### Modelos de Machine Learning
- **Regressão Linear**: Previsão simples baseada em tendência
- **Random Forest**: Reconhecimento de padrões não-lineares
- **Engenharia de Features**: Features de lag, estatísticas móveis

#### Métodos Ensemble
- **Média de Modelos**: Combinar múltiplas previsões
- **Ensemble Ponderado**: Ponderação baseada em performance
- **Stacking**: Abordagem de meta-aprendizado

### Funcionalidades

#### Pré-processamento de Dados
- **Tratamento de Valores Ausentes**: Forward fill, interpolação, imputação por média
- **Detecção de Outliers**: Identificação estatística e visual
- **Teste de Estacionariedade**: Teste ADF e diferenciação
- **Decomposição Sazonal**: Decomposição STL

#### Visualização
- **Gráficos de Séries Temporais**: Gráficos de linha interativos com zoom
- **Gráficos de Previsão**: Predições com intervalos de confiança
- **Análise de Resíduos**: Distribuição de erros e autocorrelação
- **Gráficos Sazonais**: Padrões mensais/trimestrais

#### Avaliação de Modelo
- **Métricas de Precisão**: MAE, RMSE, MAPE, SMAPE
- **Validação Cruzada**: Validação com divisão de séries temporais
- **Análise de Resíduos**: Testes de normalidade e independência
- **Intervalos de Previsão**: Intervalos de confiança e predição

### Datasets de Exemplo
- **Preços de Ações**: Dados diários do mercado de ações
- **Dados de Vendas**: Vendas mensais no varejo
- **Clima**: Temperatura e precipitação
- **Indicadores Econômicos**: PIB, inflação, desemprego

### Configuração
Personalize parâmetros de previsão:
```python
config = {
    'arima_order': (1, 1, 1),
    'seasonal_order': (1, 1, 1, 12),
    'forecast_periods': 30,
    'confidence_level': 0.95
}
```

### Dicas de Performance
- **Qualidade dos Dados**: Dados de séries temporais limpos e consistentes
- **Seleção de Modelo**: Use validação cruzada para comparação de modelos
- **Sazonalidade**: Considere padrões sazonais em seus dados
- **Horizonte de Previsão**: Previsões mais curtas são geralmente mais precisas

### Contribuindo
1. Faça um fork do repositório
2. Crie uma branch de feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adicionar nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Crie um Pull Request

### Licença
Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

