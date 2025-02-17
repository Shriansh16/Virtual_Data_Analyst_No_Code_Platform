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
import io
import plotly.io as pio

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
                    basic_stats = calculate_basic_stats(numeric_data)
                    st.dataframe(basic_stats)

                with col2:
                    st.write("#### 📈 Quartile Statistics")
                    quartile_stats = calculate_quartile_stats(numeric_data)
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
            correlation_matrix = numeric_data.corr()
            correlation_heatmap = plot_correlation_heatmap(correlation_matrix)
            st.plotly_chart(correlation_heatmap, use_container_width=True)

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
            time_series_plot = plot_time_series(data, date_col, numeric_col, window_size)
            st.plotly_chart(time_series_plot, use_container_width=True)

        # Distribution Analysis
        st.subheader("📊 Distribution Analysis")
        col1, col2 = st.columns(2)

        with col1:
            selected_feature = st.selectbox("Select Feature for Distribution", data.columns)
            distribution_plot = plot_distribution(data, selected_feature)
            st.plotly_chart(distribution_plot, use_container_width=True)

        with col2:
            if len(numeric_data.columns) >= 2:
                x_feature = st.selectbox("Select X Feature for Scatter Plot", numeric_data.columns)
                y_feature = st.selectbox("Select Y Feature for Scatter Plot",
                                       [col for col in numeric_data.columns if col != x_feature])
                scatter_plot = plot_scatter(data, x_feature, y_feature)
                st.plotly_chart(scatter_plot, use_container_width=True)

        # Outlier Analysis
        st.subheader("🔍 Outlier Analysis")
        if not numeric_data.empty:
            outlier_feature = st.selectbox("Select Feature for Outlier Analysis", numeric_data.columns)
            boxplot = plot_boxplot(data, outlier_feature)
            st.plotly_chart(boxplot, use_container_width=True)

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

    # Export to Excel Button
    if st.button("Export All Analysis to Excel"):
        with st.spinner("Exporting data to Excel..."):
            # Create a BytesIO buffer to store the Excel file
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Save Data Overview
                data.to_excel(writer, sheet_name='Data Overview', index=False)
                
                # Save Basic Statistics
                basic_stats.to_excel(writer, sheet_name='Basic Statistics', index=False)
                
                # Save Quartile Statistics
                quartile_stats.to_excel(writer, sheet_name='Quartile Statistics', index=False)
                
                # Save Shape Statistics
                shape_stats.to_excel(writer, sheet_name='Shape Statistics', index=False)
                
                # Save Normality Statistics
                normality_stats.to_excel(writer, sheet_name='Normality Statistics', index=False)
                
                # Save Correlation Analysis
                correlation_matrix.to_excel(writer, sheet_name='Correlation Analysis', index=False)
                # Save Correlation Heatmap
                correlation_heatmap.write_image("correlation_heatmap.png")
                worksheet = writer.sheets['Correlation Analysis']
                worksheet.insert_image('H1', 'correlation_heatmap.png')
                
                # Save Strong Correlations
                strong_corr_df.to_excel(writer, sheet_name='Strong Correlations', index=False)
                
                # Save Time Series Analysis
                if len(date_columns) > 0:
                    time_series_data = data[[date_col, numeric_col]]
                    time_series_data.to_excel(writer, sheet_name='Time Series Analysis', index=False)
                    # Save Time Series Plot
                    time_series_plot.write_image("time_series_plot.png")
                    worksheet = writer.sheets['Time Series Analysis']
                    worksheet.insert_image('H1', 'time_series_plot.png')
                
                # Save Distribution Analysis
                distribution_data = data[[selected_feature]]
                distribution_data.to_excel(writer, sheet_name='Distribution Analysis', index=False)
                # Save Distribution Plot
                distribution_plot.write_image("distribution_plot.png")
                worksheet = writer.sheets['Distribution Analysis']
                worksheet.insert_image('H1', 'distribution_plot.png')
                
                # Save Scatter Plot Analysis
                if len(numeric_data.columns) >= 2:
                    scatter_data = data[[x_feature, y_feature]]
                    scatter_data.to_excel(writer, sheet_name='Scatter Plot Analysis', index=False)
                    # Save Scatter Plot
                    scatter_plot.write_image("scatter_plot.png")
                    worksheet = writer.sheets['Scatter Plot Analysis']
                    worksheet.insert_image('H1', 'scatter_plot.png')
                
                # Save Outlier Analysis
                outlier_data = data[[outlier_feature]]
                outlier_data.to_excel(writer, sheet_name='Outlier Analysis', index=False)
                # Save Boxplot
                boxplot.write_image("boxplot.png")
                worksheet = writer.sheets['Outlier Analysis']
                worksheet.insert_image('H1', 'boxplot.png')

            # Get the Excel file from the buffer
            output.seek(0)
            st.success("Export completed successfully!")
            st.download_button(
                label="Download Excel File",
                data=output,
                file_name="data_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

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