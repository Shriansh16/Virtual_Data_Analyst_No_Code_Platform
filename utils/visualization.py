import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_correlation_heatmap(correlation_matrix):
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(correlation_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
    ))
    fig.update_layout(
        title='Interactive Correlation Heatmap',
        height=600,
        width=800,
        xaxis_title="Features",
        yaxis_title="Features",
    )
    return fig

def plot_time_series(data, date_col, numeric_col, window_size=None):
    if window_size:
        data['MA'] = data[numeric_col].rolling(window=window_size).mean()
        fig = px.line(data, x=date_col, y=[numeric_col, 'MA'],
                      title=f"{numeric_col} with {window_size}-period Moving Average",
                      labels={'value': 'Value', 'variable': 'Metric'})
    else:
        fig = px.line(data, x=date_col, y=numeric_col, title=f"{numeric_col} Over Time")
    return fig

def plot_distribution(data, selected_feature):
    if data[selected_feature].dtype in ['float64', 'int64']:
        fig = px.histogram(data, x=selected_feature, marginal="box",
                           title=f"Distribution of {selected_feature}")
    else:
        value_counts = data[selected_feature].value_counts().reset_index()
        value_counts.columns = [selected_feature, 'count']
        fig = px.bar(value_counts, x=selected_feature, y='count',
                      title=f"Distribution of {selected_feature}")
    return fig

def plot_scatter(data, x_feature, y_feature):
    fig = px.scatter(data, x=x_feature, y=y_feature,
                     trendline="ols",
                     title=f"Scatter Plot: {x_feature} vs {y_feature}")
    return fig

def plot_boxplot(data, feature):
    fig = go.Figure()
    fig.add_trace(go.Box(y=data[feature], name=feature))
    fig.update_layout(title=f"Box Plot: {feature}")
    return fig