import pandas as pd
import numpy as np

# API Keys and Model Configuration
GROQ_API_KEY = "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"
MODEL_NAME = "llama-3.3-70b-versatile"
MODEL_TEMPERATURE = 0.5

# Sample Datasets
SAMPLE_DATASETS = {
    "Sales Data": {
        'Date': pd.date_range(start='2024-01-01', periods=100),
        'Daily_Sales': np.random.normal(1000, 200, 100).round(2),
        'Units_Sold': np.random.randint(50, 200, 100),
        'Average_Price': np.random.normal(50, 10, 100).round(2),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 100),
        'Customer_Satisfaction': np.random.randint(1, 6, 100)
    },
    "Customer Feedback": {
        'Date': pd.date_range(start='2024-01-01', periods=100),
        'Rating': np.random.randint(1, 6, 100),
        'Response_Time_Hours': np.random.normal(24, 5, 100).round(2),
        'Resolution_Time_Hours': np.random.normal(48, 10, 100).round(2),
        'Category': np.random.choice(['Product', 'Service', 'Support', 'Billing'], 100),
        'Priority': np.random.choice(['High', 'Medium', 'Low'], 100),
        'Customer_Type': np.random.choice(['New', 'Returning', 'Premium'], 100),
        'Satisfaction_Score': np.random.randint(60, 100, 100)
    },
    "Stock Prices": {
        'Date': pd.date_range(start='2024-01-01', periods=100),
        'Open_Price': np.random.normal(100, 10, 100).round(2),
        'Close_Price': np.random.normal(100, 10, 100).round(2),
        'High_Price': np.random.normal(105, 10, 100).round(2),
        'Low_Price': np.random.normal(95, 10, 100).round(2),
        'Volume': np.random.normal(10000, 2000, 100).round(0),
        'Market_Cap': np.random.normal(1000000, 200000, 100).round(0),
        'Sector': np.random.choice(['Technology', 'Finance', 'Healthcare', 'Energy'], 100),
        'Trading_Type': np.random.choice(['Regular', 'After-Hours', 'Pre-Market'], 100)
    }
}