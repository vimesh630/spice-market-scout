import pandas as pd
import numpy as np
import random
from datetime import date, timedelta

def generate_mock_data():
    """Generates mock spice price data for testing."""
    start_date = date(2024, 1, 1)
    end_date = date.today()
    delta = end_date - start_date
    dates = [start_date + timedelta(days=i) for i in range(delta.days + 1)]

    data = {
        'Date': dates,
        'Cinnamon_Grade_ALBA': [random.randint(2500, 3000) for _ in range(len(dates))],
        'Cinnamon_Grade_C5': [random.randint(1800, 2200) for _ in range(len(dates))],
        'Market_Sentiment': [random.choice(['Bullish', 'Bearish', 'Neutral']) for _ in range(len(dates))]
    }

    df = pd.DataFrame(data)
    
    output_path = 'data/processed/spice_prices.csv'
    df.to_csv(output_path, index=False)
    print(f"Mock data generated and saved to {output_path}")

if __name__ == "__main__":
    generate_mock_data()
