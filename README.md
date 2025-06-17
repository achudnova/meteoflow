# METEOFLOW

<table>
  <tr>
    <td valign="top"><img src="pics/meteoflow-logo.png" alt="MeteoFlow Logo" width="200"/></td>
    <td valign="top">MeteoFlow is a machine learning-based weather forecasting CLI application that predicts average temperature and wind speed for Berlin. This project implements an end-to-end ML pipeline, from data collection and processing to model training and prediction.</td>
  </tr>
</table>

MeteoFlow is a machine learning-based weather forecasting CLI application that predicts average temperature and wind speed for Berlin.

This project implements an end-to-end ML pipeline:

## Key Features / Workflow

- Automated Data Collection
- Spatial Intepolation (IDW)
- Exploratory Data Analysis with visualizations
- Data Preprocessing
- Feature Engineering with time-based features and lag variables
- Model training: RandomForest and XGBoost
- Evaluation: assesses model performance using standard regression metrics (MAE, RMSE, R²)
- Prediction

<p align="center">
  <img src="pics/ml-flow.png" alt="Machine Learning Workflow"/>
</p>


## Tools & Technologies

-   **Python**
-   **Pandas** for data manipulation
-   **Scikit-learn** for machine learning models and metrics
-   **XGBoost** for the gradient boosting model
-   **Matplotlib / Seaborn** for data visualization
-   **Click / Argparse** for the CL

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/achudnova/meteoflow.git
    cd meteoflow
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage (train the models)

To run the full pipeline, execute:

```bash
python3 src/main.py 
```
