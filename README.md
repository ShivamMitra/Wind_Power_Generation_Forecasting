# Wind Power Generation Forecasting

This repository contains a comprehensive analysis and forecasting system for wind power generation data. The project includes exploratory data analysis (EDA), data preprocessing, and machine learning models for forecasting wind power generation.

## Project Overview

The project is structured into two main components:

1. **Exploratory Data Analysis (EDA)**: Understanding the wind power generation data, identifying patterns, and extracting insights.
2. **Machine Learning Forecasting**: Building and evaluating multiple models to predict wind power generation.

## Repository Structure

```
wind-power-forecasting/
├── data/                      # Data files
│   ├── merged_locations.csv   # Merged dataset
│   └── ...                    # Individual location files
├── wind_power_forecasting.ipynb # Jupyter notebook for EDA/ML forecasting 
├── models/                    # Saved models
├── results/                   # Model evaluation results
├── visualizations/            # Generated visualizations
└── README.md                  # This file
```

## Data Description

The dataset contains wind power generation data from multiple locations, including:
- Power generation values
- Wind speed at different heights (10m, 100m)
- Wind direction at different heights
- Temperature and humidity measurements
- Timestamp information

## Exploratory Data Analysis (EDA)

The EDA process is documented in `wind_power_forecasting.ipynb` and includes:

1. **Data Loading and Preprocessing**
   - Loading the merged dataset
   - Basic data information
   - Time feature extraction
   - Missing value analysis

2. **Power Generation Analysis**
   - Overall power distribution
   - Power distribution by location
   - Power statistics by location

3. **Wind Characteristics Analysis**
   - Wind speed vs. Power relationships
   - Wind direction vs. Power relationships
   - Correlation analysis of wind features

4. **Temporal Patterns Analysis**
   - Hourly patterns
   - Weekday vs. Weekend patterns
   - Monthly patterns

5. **Weather Impact Analysis**
   - Temperature vs. Power relationships
   - Humidity vs. Power relationships
   - Weather feature correlations

6. **Location-Specific Analysis**
   - Detailed analysis for each location
   - Location-specific patterns and characteristics

7. **Power Threshold Analysis**
   - Analysis of time spent above different power thresholds
   - Location comparison for power thresholds

8. **Correlation Analysis**
   - Correlation heatmap of all features
   - Top correlations with power generation
   - Strong correlation identification

## Machine Learning Forecasting

The forecasting process is implemented in `wind_power_forecasting.ipynb` and includes:

1. **Data Preprocessing**
   - Loading and cleaning data
   - Feature engineering
   - Handling categorical variables
   - Scaling numerical features

2. **Model Selection**
   - Training multiple models:
     - Linear Regression
     - Random Forest
     - Gradient Boosting
     - XGBoost
   - Evaluating models on validation set
   - Selecting the best performing model

3. **Hyperparameter Tuning**
   - Grid search for optimal hyperparameters
   - Time series cross-validation
   - Model refinement

4. **Model Evaluation**
   - Performance metrics (RMSE, MAE, R²)
   - Actual vs. Predicted visualization
   - Residual analysis
   - Feature importance analysis

5. **Model Deployment**
   - Saving the trained model
   - Generating model report
   - Visualizing results

## Requirements

To run the project, you need the following Python packages:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
jupyter
joblib
```

You can install these packages using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter joblib
```

## Usage

### Running the EDA

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open `wind_power_forecasting.ipynb` and run the cells in sequence


## Key Findings

### EDA Findings
- Patterns in power generation across different time scales
- Relationships between wind characteristics and power output
- Impact of weather conditions on power generation
- Location-specific characteristics and patterns

### Forecasting Findings
- The best performing model for wind power forecasting
- Important features for prediction
- Model performance metrics
- Recommendations for improving forecasting accuracy

## Future Work

1. **Data Collection**
   - Collect more historical data
   - Include additional weather features
   - Incorporate seasonal variations

2. **Model Improvements**
   - Experiment with deep learning models
   - Implement ensemble methods
   - Develop location-specific models

3. **System Integration**
   - Build a real-time prediction system
   - Create a web interface for predictions
   - Implement automated model retraining

## Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a feature branch
3. Making your changes
4. Submitting a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
