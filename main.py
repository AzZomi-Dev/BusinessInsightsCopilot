import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed
np.random.seed(42)

# Generate dates
dates = pd.date_range(start="2024-01-01", end="2024-06-01", freq='D')

# Define sample fields
products = ['Product A', 'Product B', 'Product C']
regions = ['North', 'South', 'East', 'West']
channels = ['Online', 'Retail', 'Partner']
ticket_categories = ['Login Issue', 'Payment Failure', 'Bug Report', 'Feature Request', 'Other']

# Generate Sales Data
sales_data = []
for date in dates:
    for _ in range(np.random.randint(3, 7)):
        sales_data.append({
            'date': date,
            'product': random.choice(products),
            'region': random.choice(regions),
            'sales_channel': random.choice(channels),
            'sales_amount': round(np.random.normal(500, 150), 2)
        })
sales_df = pd.DataFrame(sales_data)

# Generate Support Data
support_data = []
for date in dates:
    for _ in range(np.random.randint(2, 5)):
        resolution_time = max(0.5, np.random.normal(2, 1.2))
        support_data.append({
            'ticket_id': f'TK{random.randint(1000, 9999)}',
            'date': date,
            'category': random.choice(ticket_categories),
            'resolution_time': round(resolution_time, 2),
            'customer_satisfaction': random.choice([1, 2, 3, 4, 5])
        })
support_df = pd.DataFrame(support_data)

# Preview
print("Sales Data Sample:")
print(sales_df.head())

print("\nSupport Tickets Sample:")
print(support_df.head())

# Optional: Save to CSV
sales_df.to_csv("sample_sales_data.csv", index=False)
support_df.to_csv("sample_support_data.csv", index=False)
