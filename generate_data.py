import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_hotel_data(start_date, end_date, filename='hotel_demand_data.csv'):
    # Generate date range efficiently
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(date_range)
    
    # Pre-allocate arrays for efficiency
    data = {
        'date': date_range,
        'day_of_week': np.empty(n_days, dtype=int),
        'is_weekend': np.empty(n_days, dtype=int),
        'month': np.empty(n_days, dtype=int),
        'day_of_year': np.empty(n_days, dtype=int),
    }
    
    # Vectorized operations for efficiency
    data['day_of_week'] = date_range.dayofweek.values
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
    data['month'] = date_range.month.values
    data['day_of_year'] = date_range.dayofyear.values
    
    # Create realistic seasonal patterns with minimal computation
    seasonal_factor = 0.7 * np.sin(2 * np.pi * (data['day_of_year'] - 80) / 365.25) + 1
    
    # Efficient holiday and event patterns
    data['holiday_season'] = np.isin(data['month'], [11, 12]).astype(int)  # Nov-Dec holidays
    data['special_event'] = ((data['month'] == 7) & (data['day_of_week'] == 5)).astype(int)  # Summer Saturdays
    
    # Create target variable with realistic patterns
    base_demand = 150
    weekend_boost = data['is_weekend'] * 50
    holiday_boost = data['holiday_season'] * 40
    event_boost = data['special_event'] * 60
    seasonal_effect = seasonal_factor * 70
    
    # Add noise
    noise = np.random.normal(0, 20, n_days)
    
    # Combine all factors
    data['rooms_booked'] = (base_demand + weekend_boost + holiday_boost + 
                           event_boost + seasonal_effect + noise).astype(int)
    
    # Ensure rooms booked is within reasonable bounds
    data['rooms_booked'] = np.clip(data['rooms_booked'], 50, 300)
    
    # Create DataFrame efficiently
    df = pd.DataFrame(data)
    
    # Select only essential columns to reduce memory
    df = df[['date', 'day_of_week', 'is_weekend', 'holiday_season', 
             'special_event', 'day_of_year', 'month', 'rooms_booked']]
    
    df.to_csv(filename, index=False)
    print(f'Efficient realistic data generated and saved to {filename}')
    return df

if __name__ == '__main__':
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    generate_hotel_data(start_date, end_date)