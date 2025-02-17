import streamlit as st
from scipy import stats
import pandas as pd
from utils.data_loading import load_data
from utils.visualization import plot_correlation_heatmap, plot_time_series, plot_distribution, plot_scatter, plot_boxplot
from utils.statistical_analysis import calculate_basic_stats, calculate_quartile_stats, calculate_strong_correlations
from utils.ai_assistant import init_model, generate_analysis
from datetime import datetime
from config import SAMPLE_DATASETS
from PIL import Image
import os

# Page configuration
st.set_page_config(page_title="Virtual Data Analyst", layout="wide")

# Load custom CSS
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize LLM model
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

data = load_data(uploaded_file, sample_data)

if data is not None:
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
        numeric_data = data.select_dtypes(include=['float64', 'int64'])

        if not numeric_data.empty:
            st.write("### 📈 Detailed Summary Statistics")
            stats_tab1, stats_tab2 = st.tabs(["Basic Statistics", "Advanced Statistics"])

            with stats_tab1:
                col1, col2 = st.columns(2)

                with col1:
                    st.write("#### 📊 Basic Statistics")
                    st.dataframe(calculate_basic_stats(numeric_data))

                with col2:
                    st.write("#### 📈 Quartile Statistics")
                    st.dataframe(calculate_quartile_stats(numeric_data))

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
            correlation_matrix = numeric_data.corr()
            st.plotly_chart(plot_correlation_heatmap(correlation_matrix), use_container_width=True)

            # Strong Correlations
            st.write("#### 💪 Strong Correlations")
            strong_corr_df = calculate_strong_correlations(correlation_matrix)
            if not strong_corr_df.empty:
                st.dataframe(strong_corr_df)
            else:
                st.info("No strong correlations (|r| > 0.5) found between features.")

    with tab3:
        st.subheader("🔍 Deep Analysis")
        date_columns = data.select_dtypes(include=['datetime64']).columns

        if len(date_columns) > 0:
            st.subheader("📅 Time Series Analysis")
            date_col = st.selectbox("Select Date Column", date_columns)
            numeric_col = st.selectbox("Select Value Column", data.select_dtypes(include=['float64', 'int64']).columns)

            window_size = st.slider("Moving Average Window Size", min_value=2, max_value=30, value=7)
            st.plotly_chart(plot_time_series(data, date_col, numeric_col, window_size), use_container_width=True)

        # Distribution Analysis
        st.subheader("📊 Distribution Analysis")
        col1, col2 = st.columns(2)

        with col1:
            selected_feature = st.selectbox("Select Feature for Distribution", data.columns)
            st.plotly_chart(plot_distribution(data, selected_feature), use_container_width=True)

        with col2:
            if len(numeric_data.columns) >= 2:
                x_feature = st.selectbox("Select X Feature for Scatter Plot", numeric_data.columns)
                y_feature = st.selectbox("Select Y Feature for Scatter Plot",
                                       [col for col in numeric_data.columns if col != x_feature])
                st.plotly_chart(plot_scatter(data, x_feature, y_feature), use_container_width=True)

        # Outlier Analysis
        st.subheader("🔍 Outlier Analysis")
        if not numeric_data.empty:
            outlier_feature = st.selectbox("Select Feature for Outlier Analysis", numeric_data.columns)
            st.plotly_chart(plot_boxplot(data, outlier_feature), use_container_width=True)

    with tab4:
      st.subheader("💡 AI Analysis Assistant")
      prompt = st.text_area("Ask your question about the data:")

      if st.button("Generate Analysis"):
          with st.spinner("🤔 Analyzing data..."):
              response = generate_analysis(data, prompt, model)

              # Update session state
              st.session_state.analysis_count += 1
              st.session_state.last_analysis = datetime.now().strftime("%H:%M:%S")

              # Display response
              st.markdown("### 📑 Analysis Results")
              if isinstance(response, str) and response.endswith(('.png', '.jpg', '.jpeg')):
                    if os.path.exists(response):  # Check if the file exists
                        # Open the image file
                        image = Image.open(response)
                        # Display the image in Streamlit
                        st.image(image, caption="AI-Generated Graph", use_container_width=True)  # Updated parameter
                    else:
                        st.error(f"⚠️ Image file not found at: {response}")
              else:
                    # If the response is not an image, display it as text
                    st.write(response)

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