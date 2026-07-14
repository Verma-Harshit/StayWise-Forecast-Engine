# Hotel Demand Forecasting AI Model

This project implements a complete demand forecasting system for the hotel sector using machine learning. The system includes data generation, model training, testing, and a web application interface.

## Project Structure

- `generate_data.py` - Script to generate synthetic hotel demand data
- `hotel_demand_data.csv` - Generated dataset with hotel booking information
- `train_model.py` - Script to train the demand forecasting model
- `train_model.ipynb` - Jupyter notebook for model training
- `test_model.py` - Script to test and evaluate the model
- `test_model.ipynb` - Jupyter notebook for model testing
- `hotel_demand_model.pkl` - Trained machine learning model
- `app.py` - Streamlit web application for demand forecasting
- `README.md` - This documentation file

## Features

### Data Generation
- Generates synthetic hotel demand data for 2 years (2023-2024)
- Includes features like day of week, weekend indicator, holiday season, special events
- Creates realistic room booking patterns

### Machine Learning Model
- Uses Linear Regression for demand forecasting
- Features used for prediction:
  - Day of the week (0-6, where 0 is Monday)
  - Weekend indicator (Saturday/Sunday)
  - Holiday season indicator
  - Special event indicator
  - Day of the year (1-365/366)

### Web Application
- Interactive Streamlit interface
- Single prediction mode with date picker and parameter selection
- Batch prediction mode for CSV file uploads
- Real-time demand forecasting
- Input summary and model information

## Installation

1. Install required dependencies:
```bash
pip install pandas scikit-learn streamlit joblib numpy
```

2. Generate the training data:
```bash
python3 generate_data.py
```

3. Train the model:
```bash
python3 train_model.py
```

4. Test the model:
```bash
python3 test_model.py
```

5. Run the web application:
```bash
streamlit run app.py
```

## Usage

### Running the Web Application
1. Start the Streamlit app: `streamlit run app.py`
2. Open your browser to the provided URL (typically http://localhost:8501)
3. Select a date and configure parameters
4. Click "Predict Demand" to get forecasted room demand

### Batch Predictions
1. Prepare a CSV file with columns: `day_of_week`, `is_weekend`, `holiday_season`, `special_event`, `day_of_year`
2. Upload the file using the batch prediction feature
3. Download the results with predictions added

## Model Performance
The model evaluation shows:
- Mean Squared Error: 5090.49
- R-squared: -0.00

Note: The current model uses synthetic data and may require real hotel data for better performance in production environments.

## Future Improvements
- Incorporate real hotel booking data
- Add more sophisticated features (weather, local events, competitor pricing)
- Implement time series forecasting models (ARIMA, LSTM)
- Add seasonal decomposition and trend analysis
- Include external factors like economic indicators

## Live Demo
The application is deployed and accessible at: https://8501-iyicpjupa74a8gikqq5eq-82d3f405.manusvm.computer

