# Time Series Forecasting Suite

[English](#english) | [Português](#português)

## English

### Overview
Comprehensive time series forecasting suite with multiple algorithms, interactive visualizations, and a professional web interface. This toolkit is designed for accurate time series prediction and analysis, featuring robust models like ARIMA, Exponential Smoothing, Linear Regression, and Random Forest, alongside advanced analytical capabilities.

### Screenshots

#### Main Dashboard
![Time Series Forecasting Dashboard](https://github.com/user-attachments/assets/42109e9c-457f-49fd-a212-32107070d37e)

*Interactive Streamlit dashboard showing data overview and time series components*

#### Forecast Results with Model Comparison
![Forecast Results and Model Comparison](https://github.com/user-attachments/assets/e7d90673-15db-4c19-a96b-459510dff682)

*Complete forecast visualization with performance metrics, data export options, and model comparison charts*

### Features
- **Multiple Algorithms**: Implements ARIMA, Exponential Smoothing, Linear Regression, and Random Forest for diverse forecasting needs.
- **Interactive Dashboard**: A user-friendly web interface built with Streamlit, offering real-time updates and interactive controls.
- **Data Visualization**: Utilizes Plotly for professional, interactive charts, enabling detailed exploration of historical data and forecasts.
- **Forecast Accuracy Metrics**: Provides key evaluation metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for model performance assessment.
- **Seasonal Decomposition**: Analyzes time series into trend, seasonal, and residual components to understand underlying patterns.
- **Data Handling**: Supports uploading custom CSV files and includes a sample dataset for immediate use.
- **Export Results**: Allows downloading of forecast data and visualizations.
- **Model Comparison**: Facilitates side-by-side analysis of different algorithm performances.

### Technologies Used
- **Python 3.8+**
- **Streamlit**: For building the interactive web application.
- **Pandas**: Essential for data manipulation and analysis.
- **NumPy**: Provides powerful numerical computing capabilities.
- **Scikit-learn**: Offers various machine learning algorithms, including Linear Regression and RandomForestRegressor.
- **Plotly**: For creating rich, interactive data visualizations.
- **Statsmodels**: For statistical modeling, including ARIMA and Exponential Smoothing.

### Installation

To set up the Time Series Forecasting Suite, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/galafis/Time-Series-Forecasting-Suite.git
    cd Time-Series-Forecasting-Suite
    ```

2.  **Install dependencies**:
    Ensure you have `pip` installed. Then, install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**:
    Launch the Streamlit application from your terminal:
    ```bash
    streamlit run forecasting_suite.py
    ```

4.  **Access the application**:
    Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501` or `http://localhost:8502`).

### Usage

#### Web Interface
Once the application is running, you can interact with it through your web browser:

1.  **Data Input**: Choose to use the provided sample data or upload your own CSV file. If uploading, select the appropriate date and value columns.
2.  **Data Exploration**: Review the data overview, recent values, and time series components plots to understand your data.
3.  **Model Selection**: From the sidebar, select your preferred forecasting algorithm (e.g., Random Forest, ARIMA).
4.  **Generate Forecast**: Adjust the number of forecast days using the slider and click the "🚀 Train Models & Generate Forecast" button.
5.  **Evaluate Results**: Observe the model performance metrics (MAE, RMSE) and the interactive forecast plot.
6.  **Export**: Download the forecast data and charts as needed.

#### Using Example Datasets
The repository includes three example datasets in the `examples/` directory:

```bash
# From the Streamlit UI, upload one of these files:
# - examples/sales_data.csv (retail sales with seasonality)
# - examples/temperature_data.csv (weather data)
# - examples/stock_price_data.csv (financial data)
```

#### Programmatic Usage
You can also use the forecasting suite programmatically:

```python
import pandas as pd
from forecasting_suite import TimeSeriesForecaster

# Load your data
data = pd.read_csv('your_data.csv')

# Initialize forecaster
forecaster = TimeSeriesForecaster()
forecaster.load_data(data, date_column='date', value_column='value')

# Train models
models, metrics = forecaster.train_models()

# Generate forecast
forecast_rf = forecaster.forecast(days=30, model_name='Random Forest')
forecast_arima = forecaster.forecast(days=30, model_name='ARIMA')

# View metrics
print(metrics)

# Create visualizations
fig = forecaster.create_forecast_plot('Random Forest')
fig.show()
```

### CSV Data Format
Your CSV file should contain at minimum:
- **Date column**: Dates in any standard format (e.g., 'YYYY-MM-DD', 'MM/DD/YYYY')
- **Value column**: Numeric values to forecast

Example CSV structure:
```csv
date,value
2023-01-01,100.5
2023-01-02,102.3
2023-01-03,98.7
...
```

### Forecasting Models

#### ARIMA (AutoRegressive Integrated Moving Average)
-   **Description**: A statistical model that uses past values to predict future ones, suitable for data with trends and seasonality.
-   **Application**: Best for stationary time series with identifiable trends and seasonal patterns.

#### Exponential Smoothing
-   **Description**: A family of models that predict future values based on weighted averages of past observations, with weights decaying exponentially as observations get older.
-   **Application**: Effective for data with trend and/or seasonal components, offering simple, double, and triple (Holt-Winters) variations.

#### Machine Learning Models
-   **Linear Regression**: A straightforward model that predicts future values by fitting a linear equation to the observed data.
-   **Random Forest**: An ensemble learning method that builds multiple decision trees and merges their predictions to improve accuracy and control overfitting. Excellent for capturing non-linear relationships.
-   **Feature Engineering**: Both Linear Regression and Random Forest models leverage engineered features such as lag values and rolling statistics to enhance prediction capabilities.

### Functionalities

#### Data Preprocessing
-   **Missing Value Handling**: Automatic interpolation (linear method) to fill gaps in the time series data.
-   **Frequency Enforcement**: Ensures data is uniformly sampled (e.g., daily frequency) for consistent analysis.

#### Visualization
-   **Interactive Time Series Plots**: Powered by Plotly, these charts allow zooming and panning for detailed data inspection.
-   **Forecast Plots**: Visual representation of predictions, showing historical data, forecasted values, and a clear demarcation of the forecast start.
-   **Time Series Components Plot**: Displays the decomposed trend, seasonal, and residual components of the time series.

#### Model Evaluation
-   **Accuracy Metrics**: Provides Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for both training and testing datasets to assess model accuracy.

### Sample Datasets
-   The application includes a built-in sample dataset for immediate demonstration and testing purposes.
-   Additional example datasets are available in the `examples/` directory for testing different scenarios.

### Troubleshooting

#### Common Issues

**Issue: "Model not trained" error**
- **Solution**: Click the "🚀 Train Models & Generate Forecast" button before trying to generate forecasts.

**Issue: ARIMA or Exponential Smoothing fails to train**
- **Solution**: These models require sufficient data (at least 2-3 seasonal periods). Try using more historical data or switch to ML models (Random Forest, Linear Regression).

**Issue: Upload fails or data not loading**
- **Solution**: Ensure your CSV file:
  - Has a proper header row
  - Uses standard date formats
  - Contains numeric values in the value column
  - Has no completely empty rows

**Issue: Poor forecast accuracy**
- **Solution**: 
  - Try different models and compare performance
  - Ensure you have enough historical data (at least 1-2 years recommended)
  - Check for data quality issues (outliers, missing values)
  - Consider the nature of your data (some patterns are harder to predict)

### FAQ

**Q: How much data do I need for accurate forecasts?**
A: Generally, at least 1-2 years of daily data is recommended. More data typically leads to better forecasts, especially for seasonal patterns.

**Q: Which model should I use?**
A: 
- **Random Forest**: Great for complex, non-linear patterns
- **ARIMA**: Best for stationary data with clear trends
- **Exponential Smoothing**: Excellent for seasonal data
- **Linear Regression**: Good baseline for simple trends

Use the Model Comparison feature to evaluate all models and choose the best performer.

**Q: Can I forecast multiple time series at once?**
A: Currently, the application handles one time series at a time. For batch processing, use the programmatic interface.

**Q: What's the maximum forecast horizon?**
A: You can forecast up to 365 days ahead, but accuracy typically decreases for longer horizons.

**Q: How are missing values handled?**
A: The system automatically interpolates missing values using linear interpolation.

**Q: Can I use this for hourly or minute-level data?**
A: Yes! The system automatically detects your data frequency. Just ensure your date column includes time information.

### Performance Considerations

- **Training Time**: Depends on data size and model complexity
  - Linear Regression: Fast (~1-2 seconds)
  - Random Forest: Moderate (~5-10 seconds)
  - ARIMA: Can be slow for large datasets (~10-30 seconds)
  - Exponential Smoothing: Moderate (~5-15 seconds)

- **Memory Usage**: Approximately 100-500MB for typical datasets (1-5 years of daily data)

- **Optimal Data Size**: 365-1825 data points (1-5 years of daily data)

### Contributing
We welcome contributions to enhance the Time Series Forecasting Suite! Please follow these guidelines:

1.  **Fork the repository**.
2.  **Create a feature branch** (`git checkout -b feature/your-feature-name`).
3.  **Commit your changes** (`git commit -am 'Add new feature'`).
4.  **Push to the branch** (`git push origin feature/your-feature-name`).
5.  **Create a Pull Request**.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Author

**Gabriel Demetrios Lafis**
- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-lafis)

---

## Português

### Visão Geral
Suite abrangente de previsão de séries temporais com múltiplos algoritmos, visualizações interativas e uma interface web profissional. Este kit de ferramentas foi projetado para previsão e análise precisas de séries temporais, apresentando modelos robustos como ARIMA, Suavização Exponencial, Regressão Linear e Random Forest, juntamente com capacidades analíticas avançadas.

### Funcionalidades
-   **Múltiplos Algoritmos**: Implementa ARIMA, Suavização Exponencial, Regressão Linear e Random Forest para diversas necessidades de previsão.
-   **Dashboard Interativo**: Uma interface web amigável construída com Streamlit, oferecendo atualizações em tempo real e controles interativos.
-   **Visualização de Dados**: Utiliza Plotly para gráficos profissionais e interativos, permitindo a exploração detalhada de dados históricos e previsões.
-   **Métricas de Precisão de Previsão**: Fornece métricas de avaliação chave, como Erro Absoluto Médio (MAE) e Raiz do Erro Quadrático Médio (RMSE) para avaliação do desempenho do modelo.
-   **Decomposição Sazonal**: Analisa séries temporais em componentes de tendência, sazonalidade e resíduos para entender padrões subjacentes.
-   **Manipulação de Dados**: Suporta o upload de arquivos CSV personalizados e inclui um conjunto de dados de exemplo para uso imediato.
-   **Exportar Resultados**: Permite o download de dados de previsão e visualizações.
-   **Comparação de Modelos**: Facilita a análise lado a lado do desempenho de diferentes algoritmos.

### Tecnologias Utilizadas
-   **Python 3.8+**
-   **Streamlit**: Para a construção da aplicação web interativa.
-   **Pandas**: Essencial para manipulação e análise de dados.
-   **NumPy**: Fornece poderosas capacidades de computação numérica.
-   **Scikit-learn**: Oferece vários algoritmos de machine learning, incluindo Regressão Linear e RandomForestRegressor.
-   **Plotly**: Para a criação de visualizações de dados ricas e interativas.
-   **Statsmodels**: Para modelagem estatística, incluindo ARIMA e Suavização Exponencial.

### Instalação

Para configurar a Time Series Forecasting Suite, siga estes passos:

1.  **Clone o repositório**:
    ```bash
    git clone https://github.com/galafis/Time-Series-Forecasting-Suite.git
    cd Time-Series-Forecasting-Suite
    ```

2.  **Instale as dependências**:
    Certifique-se de ter o `pip` instalado. Em seguida, instale os pacotes Python necessários:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Execute a aplicação**:
    Inicie a aplicação Streamlit a partir do seu terminal:
    ```bash
    streamlit run forecasting_suite.py
    ```

4.  **Execute os testes**:
    ```bash
    python3 -m pytest test_forecasting_suite.py
    ```

5.  **Acesse a aplicação**:
    Abra seu navegador e navegue para a URL local fornecida pelo Streamlit (geralmente `http://localhost:8501` ou `http://localhost:8502`).

### Uso

#### Interface Web
Uma vez que a aplicação esteja em execução, você pode interagir com ela através do seu navegador web:

1.  **Entrada de Dados**: Escolha usar os dados de exemplo fornecidos ou faça o upload do seu próprio arquivo CSV. Se estiver fazendo o upload, selecione as colunas de data e valor apropriadas.
2.  **Exploração de Dados**: Revise a visão geral dos dados, valores recentes e gráficos de componentes da série temporal para entender seus dados.
3.  **Seleção de Modelo**: Na barra lateral, selecione seu algoritmo de previsão preferido (por exemplo, Random Forest, ARIMA).
4.  **Gerar Previsão**: Ajuste o número de dias de previsão usando o controle deslizante e clique no botão "🚀 Treinar Modelos e Gerar Previsão".
5.  **Avaliar Resultados**: Observe as métricas de desempenho do modelo (MAE, RMSE) e o gráfico de previsão interativo.
6.  **Exportar**: Baixe os dados de previsão e os gráficos conforme necessário.

#### Usando Datasets de Exemplo
O repositório inclui três conjuntos de dados de exemplo no diretório `examples/`:

```bash
# Na interface Streamlit, faça upload de um destes arquivos:
# - examples/sales_data.csv (vendas com sazonalidade)
# - examples/temperature_data.csv (dados meteorológicos)
# - examples/stock_price_data.csv (dados financeiros)
```

#### Uso Programático
Você também pode usar o suite de previsão programaticamente:

```python
import pandas as pd
from forecasting_suite import TimeSeriesForecaster

# Carregue seus dados
data = pd.read_csv('seus_dados.csv')

# Inicialize o forecaster
forecaster = TimeSeriesForecaster()
forecaster.load_data(data, date_column='date', value_column='value')

# Treine os modelos
models, metrics = forecaster.train_models()

# Gere previsões
forecast_rf = forecaster.forecast(days=30, model_name='Random Forest')
forecast_arima = forecaster.forecast(days=30, model_name='ARIMA')

# Visualize as métricas
print(metrics)

# Crie visualizações
fig = forecaster.create_forecast_plot('Random Forest')
fig.show()
```

### Formato de Dados CSV
Seu arquivo CSV deve conter no mínimo:
- **Coluna de data**: Datas em qualquer formato padrão (ex: 'YYYY-MM-DD', 'DD/MM/YYYY')
- **Coluna de valor**: Valores numéricos para prever

Exemplo de estrutura CSV:
```csv
date,value
2023-01-01,100.5
2023-01-02,102.3
2023-01-03,98.7
...
```

### Modelos de Previsão

#### ARIMA (AutoRegressive Integrated Moving Average)
-   **Descrição**: Um modelo estatístico que usa valores passados para prever futuros, adequado para dados com tendências e sazonalidade.
-   **Aplicação**: Melhor para séries temporais estacionárias com tendências e padrões sazonais identificáveis.

#### Suavização Exponencial
-   **Descrição**: Uma família de modelos que preveem valores futuros com base em médias ponderadas de observações passadas, com pesos decaindo exponencialmente à medida que as observações envelhecem.
-   **Aplicação**: Eficaz para dados com componentes de tendência e/ou sazonalidade, oferecendo variações simples, duplas e triplas (Holt-Winters).

#### Modelos de Machine Learning
-   **Regressão Linear**: Um modelo direto que prevê valores futuros ajustando uma equação linear aos dados observados.
-   **Random Forest**: Um método de aprendizado de conjunto que constrói múltiplas árvores de decisão e mescla suas previsões para melhorar a precisão e controlar o overfitting. Excelente para capturar relações não lineares.
-   **Engenharia de Features**: Ambos os modelos de Regressão Linear e Random Forest utilizam features engenheiradas, como valores de lag e estatísticas móveis, para aprimorar as capacidades de previsão.

### Funcionalidades

#### Pré-processamento de Dados
-   **Tratamento de Valores Ausentes**: Interpolação automática (método linear) para preencher lacunas nos dados da série temporal.
-   **Imposição de Frequência**: Garante que os dados sejam amostrados uniformemente (por exemplo, frequência diária) para análise consistente.

#### Visualização
-   **Gráficos Interativos de Séries Temporais**: Alimentados por Plotly, esses gráficos permitem zoom e panorâmica para inspeção detalhada dos dados.
-   **Gráficos de Previsão**: Representação visual das previsões, mostrando dados históricos, valores previstos e uma clara demarcação do início da previsão.
-   **Gráfico de Componentes da Série Temporal**: Exibe os componentes de tendência, sazonalidade e resíduos decompostos da série temporal.

#### Avaliação de Modelo
-   **Métricas de Precisão**: Fornece Erro Absoluto Médio (MAE) e Raiz do Erro Quadrático Médio (RMSE) para conjuntos de dados de treinamento e teste para avaliar a precisão do modelo.

### Datasets de Exemplo
-   A aplicação inclui um conjunto de dados de exemplo integrado para fins de demonstração e teste imediatos.
-   Conjuntos de dados de exemplo adicionais estão disponíveis no diretório `examples/` para testar diferentes cenários.

### Solução de Problemas

#### Problemas Comuns

**Problema: Erro "Model not trained"**
- **Solução**: Clique no botão "🚀 Treinar Modelos e Gerar Previsão" antes de tentar gerar previsões.

**Problema: ARIMA ou Suavização Exponencial falham ao treinar**
- **Solução**: Estes modelos requerem dados suficientes (pelo menos 2-3 períodos sazonais). Tente usar mais dados históricos ou mude para modelos ML (Random Forest, Regressão Linear).

**Problema: Falha no upload ou dados não carregam**
- **Solução**: Certifique-se que seu arquivo CSV:
  - Tem uma linha de cabeçalho adequada
  - Usa formatos de data padrão
  - Contém valores numéricos na coluna de valor
  - Não tem linhas completamente vazias

**Problema: Baixa precisão nas previsões**
- **Solução**: 
  - Experimente diferentes modelos e compare o desempenho
  - Certifique-se de ter dados históricos suficientes (pelo menos 1-2 anos recomendado)
  - Verifique problemas de qualidade de dados (outliers, valores ausentes)
  - Considere a natureza dos seus dados (alguns padrões são mais difíceis de prever)

### Perguntas Frequentes (FAQ)

**P: Quantos dados eu preciso para previsões precisas?**
R: Geralmente, pelo menos 1-2 anos de dados diários é recomendado. Mais dados tipicamente levam a melhores previsões, especialmente para padrões sazonais.

**P: Qual modelo devo usar?**
R: 
- **Random Forest**: Ótimo para padrões complexos e não-lineares
- **ARIMA**: Melhor para dados estacionários com tendências claras
- **Suavização Exponencial**: Excelente para dados sazonais
- **Regressão Linear**: Boa linha de base para tendências simples

Use o recurso de Comparação de Modelos para avaliar todos os modelos e escolher o melhor.

**P: Posso prever múltiplas séries temporais ao mesmo tempo?**
R: Atualmente, a aplicação lida com uma série temporal por vez. Para processamento em lote, use a interface programática.

**P: Qual é o horizonte máximo de previsão?**
R: Você pode prever até 365 dias à frente, mas a precisão tipicamente diminui para horizontes mais longos.

**P: Como são tratados os valores ausentes?**
R: O sistema automaticamente interpola valores ausentes usando interpolação linear.

**P: Posso usar isto para dados por hora ou por minuto?**
R: Sim! O sistema detecta automaticamente a frequência dos seus dados. Apenas certifique-se que sua coluna de data inclui informações de tempo.

### Considerações de Desempenho

- **Tempo de Treinamento**: Depende do tamanho dos dados e complexidade do modelo
  - Regressão Linear: Rápido (~1-2 segundos)
  - Random Forest: Moderado (~5-10 segundos)
  - ARIMA: Pode ser lento para grandes datasets (~10-30 segundos)
  - Suavização Exponencial: Moderado (~5-15 segundos)

- **Uso de Memória**: Aproximadamente 100-500MB para datasets típicos (1-5 anos de dados diários)

- **Tamanho Ótimo de Dados**: 365-1825 pontos de dados (1-5 anos de dados diários)

### Contribuindo
Aceitamos contribuições para aprimorar a Time Series Forecasting Suite! Por favor, siga estas diretrizes:

1.  **Faça um fork do repositório**.
2.  **Crie uma branch de feature** (`git checkout -b feature/seu-nome-da-feature`).
3.  **Commit suas mudanças** (`git commit -am 'Adicionar nova feature'`).
4.  **Push para a branch** (`git push origin feature/seu-nome-da-feature`).
5.  **Crie um Pull Request**.

### Licença
Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

### Autor

**Gabriel Demetrios Lafis**
- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-lafis)
