import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_groq import ChatGroq
from pandasai import SmartDataframe
import os
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from scipy import stats

# Page configuration
st.set_page_config(page_title="Virtual Data Analyst", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    .stats-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize LLM model
@st.cache_resource
def init_model():
    return ChatGroq(
        groq_api_key="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",
        model="llama-3.3-70b-versatile",
        temperature=0.5
    )

model = init_model()

# App Header with metrics
st.title("🔍 Virtual Data Analyst - No Code Platform")
col1, col2, col3 = st.columns(3)

# Session State initialization
if 'analysis_count' not in st.session_state:
    st.session_state.analysis_count = 0
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = None

with col1:
    st.metric("Analyses Run", st.session_state.analysis_count)
with col2:
    st.metric("Last Analysis", st.session_state.last_analysis or "None")
with col3:
    st.metric("Model", "LLaMA-3.3-70B")

# File Upload with additional options
st.subheader("📂 Data Input")
upload_col1, upload_col2 = st.columns([2, 1])

with upload_col1:
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
with upload_col2:
    sample_data = st.selectbox(
        "Or try sample dataset",
        ["None", "Sales Data", "Customer Feedback", "Stock Prices"]
    )

if uploaded_file is not None or sample_data != "None":
    # Load data based on selection
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        # Sample datasets
        sample_datasets = {
            "Sales Data": pd.DataFrame({
                'Date': pd.date_range(start='2024-01-01', periods=100),
                'Daily_Sales': np.random.normal(1000, 200, 100).round(2),
                'Units_Sold': np.random.randint(50, 200, 100),
                'Average_Price': np.random.normal(50, 10, 100).round(2),
                'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
                'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 100),
                'Customer_Satisfaction': np.random.randint(1, 6, 100)
            }),
            "Customer Feedback": pd.DataFrame({
                'Date': pd.date_range(start='2024-01-01', periods=100),
                'Rating': np.random.randint(1, 6, 100),
                'Response_Time_Hours': np.random.normal(24, 5, 100).round(2),
                'Resolution_Time_Hours': np.random.normal(48, 10, 100).round(2),
                'Category': np.random.choice(['Product', 'Service', 'Support', 'Billing'], 100),
                'Priority': np.random.choice(['High', 'Medium', 'Low'], 100),
                'Customer_Type': np.random.choice(['New', 'Returning', 'Premium'], 100),
                'Satisfaction_Score': np.random.randint(60, 100, 100)
            }),
            "Stock Prices": pd.DataFrame({
                'Date': pd.date_range(start='2024-01-01', periods=100),
                'Open_Price': np.random.normal(100, 10, 100).round(2),
                'Close_Price': np.random.normal(100, 10, 100).round(2),
                'High_Price': np.random.normal(105, 10, 100).round(2),
                'Low_Price': np.random.normal(95, 10, 100).round(2),
                'Volume': np.random.normal(10000, 2000, 100).round(0),
                'Market_Cap': np.random.normal(1000000, 200000, 100).round(0),
                'Sector': np.random.choice(['Technology', 'Finance', 'Healthcare', 'Energy'], 100),
                'Trading_Type': np.random.choice(['Regular', 'After-Hours', 'Pre-Market'], 100)
            })
        }
        data = sample_datasets[sample_data]

    # Data Overview Tab
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Statistical Analysis", "🔍 Deep Analysis", "💡 AI Assistant"])

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Data Preview")
            st.dataframe(data.head(5))

        with col2:
            st.subheader("Quick Stats")
            st.write(f"**Rows:** {data.shape[0]}")
            st.write(f"**Columns:** {data.shape[1]}")
            st.write(f"**Memory Usage:** {data.memory_usage().sum() / 1024:.2f} KB")

        # Data Quality Check
        st.subheader("Data Quality Check")
        quality_col1, quality_col2, quality_col3 = st.columns(3)

        with quality_col1:
            missing_count = data.isnull().sum().sum()
            st.metric("Missing Values", missing_count)

        with quality_col2:
            duplicate_count = data.duplicated().sum()
            st.metric("Duplicate Rows", duplicate_count)

        with quality_col3:
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
            st.metric("Numeric Columns", len(numeric_cols))

    with tab2:
        st.subheader("📊 Statistical Analysis")

        # Enhanced Summary Statistics
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        if not numeric_data.empty:
            st.write("### 📈 Detailed Summary Statistics")

            # Create tabs for different statistical views
            stats_tab1, stats_tab2 = st.tabs(["Basic Statistics", "Advanced Statistics"])

            with stats_tab1:
                col1, col2 = st.columns(2)

                with col1:
                    st.write("#### 📊 Basic Statistics")
                    basic_stats = pd.DataFrame({
                        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                        **{col: [
                            f"{numeric_data[col].mean():.2f}",
                            f"{numeric_data[col].median():.2f}",
                            f"{numeric_data[col].std():.2f}",
                            f"{numeric_data[col].min():.2f}",
                            f"{numeric_data[col].max():.2f}"
                        ] for col in numeric_data.columns}
                    }).set_index('Metric')
                    st.dataframe(basic_stats)

                with col2:
                    st.write("#### 📈 Quartile Statistics")
                    quartile_stats = pd.DataFrame({
                        'Metric': ['25th Percentile', '50th Percentile', '75th Percentile', 'IQR'],
                        **{col: [
                            f"{numeric_data[col].quantile(0.25):.2f}",
                            f"{numeric_data[col].quantile(0.50):.2f}",
                            f"{numeric_data[col].quantile(0.75):.2f}",
                            f"{(numeric_data[col].quantile(0.75) - numeric_data[col].quantile(0.25)):.2f}"
                        ] for col in numeric_data.columns}
                    }).set_index('Metric')
                    st.dataframe(quartile_stats)

            with stats_tab2:
                col1, col2 = st.columns(2)

                with col1:
                    st.write("#### 🎯 Shape Statistics")
                    shape_stats = pd.DataFrame({
                        'Metric': ['Skewness', 'Kurtosis'],
                        **{col: [
                            f"{stats.skew(numeric_data[col].dropna()):.2f}",
                            f"{stats.kurtosis(numeric_data[col].dropna()):.2f}"
                        ] for col in numeric_data.columns}
                    }).set_index('Metric')
                    st.dataframe(shape_stats)

                with col2:
                    st.write("#### 📊 Distribution Tests")
                    normality_stats = pd.DataFrame({
                        'Metric': ['Shapiro p-value'],
                        **{col: [
                            f"{stats.shapiro(numeric_data[col].dropna())[1]:.4f}"
                        ] for col in numeric_data.columns}
                    }).set_index('Metric')
                    st.dataframe(normality_stats)

            # Enhanced Correlation Analysis
            st.write("### 🔗 Correlation Analysis")

            # Correlation matrix calculation
            correlation_matrix = numeric_data.corr()

            # Interactive Correlation Heatmap with Plotly
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

            st.plotly_chart(fig, use_container_width=True)

            # Correlation Strength Analysis
            st.write("#### 💪 Strong Correlations")
            strong_correlations = []

            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i,j]
                    if abs(corr_value) > 0.5:  # Threshold for strong correlation
                        strong_correlations.append({
                            'Feature 1': correlation_matrix.columns[i],
                            'Feature 2': correlation_matrix.columns[j],
                            'Correlation': corr_value
                        })

            if strong_correlations:
                strong_corr_df = pd.DataFrame(strong_correlations)
                strong_corr_df['Correlation'] = strong_corr_df['Correlation'].round(3)
                st.dataframe(strong_corr_df)
            else:
                st.info("No strong correlations (|r| > 0.5) found between features.")

    with tab3:
        st.subheader("🔍 Deep Analysis")

        # Time Series Analysis (if date column exists)
        date_columns = data.select_dtypes(include=['datetime64']).columns
        if len(date_columns) > 0:
            st.subheader("📅 Time Series Analysis")
            date_col = st.selectbox("Select Date Column", date_columns)
            numeric_col = st.selectbox("Select Value Column", data.select_dtypes(include=['float64', 'int64']).columns)

            fig = px.line(data, x=date_col, y=numeric_col, title=f"{numeric_col} Over Time")
            st.plotly_chart(fig, use_container_width=True)

            # Add moving averages
            window_size = st.slider("Moving Average Window Size", min_value=2, max_value=30, value=7)
            data['MA'] = data[numeric_col].rolling(window=window_size).mean()

            fig = px.line(data, x=date_col, y=[numeric_col, 'MA'],
                         title=f"{numeric_col} with {window_size}-period Moving Average",
                         labels={'value': 'Value', 'variable': 'Metric'})
            st.plotly_chart(fig, use_container_width=True)

        # Distribution Analysis
        st.subheader("📊 Distribution Analysis")
        col1, col2 = st.columns(2)

        with col1:
            selected_feature = st.selectbox("Select Feature for Distribution", data.columns)
            if data[selected_feature].dtype in ['float64', 'int64']:
                fig = px.histogram(data, x=selected_feature, marginal="box",
                                 title=f"Distribution of {selected_feature}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Handle non-numeric features
                value_counts = data[selected_feature].value_counts().reset_index()
                value_counts.columns = [selected_feature, 'count']  # Rename columns
                fig = px.bar(value_counts,
                             x=selected_feature, y='count',
                             title=f"Distribution of {selected_feature}")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if len(numeric_data.columns) >= 2:
                x_feature = st.selectbox("Select X Feature for Scatter Plot", numeric_data.columns)
                y_feature = st.selectbox("Select Y Feature for Scatter Plot",
                                       [col for col in numeric_data.columns if col != x_feature])

                fig = px.scatter(data, x=x_feature, y=y_feature,
                               trendline="ols",
                               title=f"Scatter Plot: {x_feature} vs {y_feature}")
                st.plotly_chart(fig, use_container_width=True)

        # Outlier Analysis
        st.subheader("🔍 Outlier Analysis")
        if not numeric_data.empty:
            outlier_feature = st.selectbox("Select Feature for Outlier Analysis", numeric_data.columns)

            # Calculate outliers using IQR method
            Q1 = numeric_data[outlier_feature].quantile(0.25)
            Q3 = numeric_data[outlier_feature].quantile(0.75)
            IQR = Q3 - Q1
            outlier_bounds = {
                'lower': Q1 - 1.5 * IQR,
                'upper': Q3 + 1.5 * IQR
            }

            outliers = numeric_data[
                (numeric_data[outlier_feature] < outlier_bounds['lower']) |
                (numeric_data[outlier_feature] > outlier_bounds['upper'])
            ]

            fig = go.Figure()
            fig.add_trace(go.Box(y=numeric_data[outlier_feature], name=outlier_feature))
            fig.update_layout(title=f"Box Plot with Outliers: {outlier_feature}")
            st.plotly_chart(fig, use_container_width=True)

            st.write(f"Number of outliers detected: {len(outliers)}")
            if len(outliers) > 0:
                st.dataframe(outliers)

    with tab4:
        st.subheader("💡 AI Analysis Assistant")

        # Predefined Analysis Templates
        analysis_template = st.selectbox(
            "Choose Analysis Template",
            ["Custom Question", "Trend Analysis", "Anomaly Detection", "Predictive Insights",
             "Feature Importance", "Data Quality Assessment"]
        )

        if analysis_template == "Custom Question":
            prompt = st.text_area("Ask your question about the data:")
        else:
            template_prompts = {
                "Trend Analysis": "Analyze the main trends in the data, focusing on patterns and changes over time.",
                "Anomaly Detection": "Identify and explain any unusual patterns or outliers in the data.",
                "Predictive Insights": "Based on the current data patterns, what insights can you provide about future trends?",
                "Feature Importance": "Which features in the dataset appear to be most important and why?",
                "Data Quality Assessment": "Assess the quality of this dataset, including completeness, consistency, and potential issues."
            }
            prompt = template_prompts[analysis_template]
            st.info(f"Using template prompt: {prompt}")

        if st.button("Generate Analysis"):
            with st.spinner("🤔 Analyzing data..."):
                # Convert to SmartDataframe for AI analysis
                df = SmartDataframe(data, config={"llm": model})
                response = df.chat(prompt)

                # Update session state
                st.session_state.analysis_count += 1
                st.session_state.last_analysis = datetime.now().strftime("%H:%M:%S")

                # Display response
                st.markdown("### 📑 Analysis Results")
                st.write(response)

                # Generate and display relevant visualizations based on the analysis
                st.markdown("### 📊 Supporting Visualizations")

                try:
                    if len(numeric_data.columns) > 0:
                        # Add relevant visualizations based on the analysis type
                        if analysis_template == "Trend Analysis" and len(date_columns) > 0:
                            for col in numeric_data.columns[:2]:  # Show first 2 numeric columns
                                fig = px.line(data, x=date_columns[0], y=col, title=f"Trend of {col}")
                                st.plotly_chart(fig, use_container_width=True)

                        elif analysis_template == "Anomaly Detection":
                            for col in numeric_data.columns[:2]:
                                fig = go.Figure()
                                fig.add_trace(go.Box(y=data[col], name=col))
                                fig.update_layout(title=f"Distribution and Outliers: {col}")
                                st.plotly_chart(fig, use_container_width=True)

                        elif analysis_template == "Feature Importance":
                            # Show correlation heatmap
                            corr_matrix = numeric_data.corr()
                            fig = px.imshow(corr_matrix,
                                          title="Feature Correlations",
                                          color_continuous_scale="RdBu")
                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning("Could not generate additional visualizations.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🚀 Powered by LLaMA 3.3 | Made with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)