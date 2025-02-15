# utils/statistical_analysis.py

import pandas as pd
from scipy import stats

def calculate_basic_stats(numeric_data):
    return pd.DataFrame({
        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
        **{col: [
            f"{numeric_data[col].mean():.2f}",
            f"{numeric_data[col].median():.2f}",
            f"{numeric_data[col].std():.2f}",
            f"{numeric_data[col].min():.2f}",
            f"{numeric_data[col].max():.2f}"
        ] for col in numeric_data.columns}
    }).set_index('Metric')

def calculate_quartile_stats(numeric_data):
    return pd.DataFrame({
        'Metric': ['25th Percentile', '50th Percentile', '75th Percentile', 'IQR'],
        **{col: [
            f"{numeric_data[col].quantile(0.25):.2f}",
            f"{numeric_data[col].quantile(0.50):.2f}",
            f"{numeric_data[col].quantile(0.75):.2f}",
            f"{numeric_data[col].quantile(0.75) - numeric_data[col].quantile(0.25):.2f}"
        ] for col in numeric_data.columns}
    }).set_index('Metric')

def calculate_strong_correlations(correlation_matrix, threshold=0.5):
    strong_correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                strong_correlations.append({
                    'Feature 1': correlation_matrix.columns[i],
                    'Feature 2': correlation_matrix.columns[j],
                    'Correlation': corr_value
                })
    return pd.DataFrame(strong_correlations)