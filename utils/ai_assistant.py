# utils/ai_assistant.py

from langchain_groq import ChatGroq
from pandasai import SmartDataframe
from config import GROQ_API_KEY, MODEL_NAME, MODEL_TEMPERATURE
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def init_model():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model=MODEL_NAME,
        temperature=MODEL_TEMPERATURE
    )

def generate_analysis(data, prompt, model):
    df = SmartDataframe(data, config={"llm": model})
    return df.chat(prompt)

def generate_supporting_visualizations(data, analysis_type, numeric_data, date_columns):
    """Generate supporting visualizations based on the analysis type."""
    visualizations = []

    if analysis_type == "Trend Analysis" and len(date_columns) > 0:
        for col in numeric_data.columns[:2]:  # Show first 2 numeric columns
            fig = px.line(data, x=date_columns[0], y=col, title=f"Trend of {col}")
            visualizations.append(fig)

    elif analysis_type == "Anomaly Detection":
        for col in numeric_data.columns[:2]:  # Show first 2 numeric columns
            fig = go.Figure()
            fig.add_trace(go.Box(y=data[col], name=col))
            fig.update_layout(title=f"Distribution and Outliers: {col}")
            visualizations.append(fig)

    elif analysis_type == "Feature Importance":
        # Show correlation heatmap
        corr_matrix = numeric_data.corr()
        fig = px.imshow(corr_matrix,
                        title="Feature Correlations",
                        color_continuous_scale="RdBu")
        visualizations.append(fig)

    return visualizations