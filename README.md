# Virtual Data Analyst - No Code Platform

## Overview

The **Virtual Data Analyst** is a no-code platform designed to help users analyze and visualize their data effortlessly. Built with Streamlit, this application provides a wide range of statistical and visual analysis tools, including:

- **Data Overview**: Quick stats, data quality checks, and data preview.
- **Statistical Analysis**: Basic and advanced statistics, correlation analysis, and distribution tests.
- **Deep Analysis**: Time series analysis, distribution analysis, scatter plots, and outlier detection.
- **AI Assistant**: Generate insights and analysis using a powerful AI model (LLaMA-3.3-70B).

## Features

### 📂 Data Input
- Upload your own CSV or Excel file.
- Choose from sample datasets (Sales Data, Customer Feedback, Stock Prices).

### 📊 Data Overview
- Preview the first 5 rows of your dataset.
- Quick stats: number of rows, columns, and memory usage.
- Data quality checks: missing values, duplicate rows, and numeric columns.

### 📈 Statistical Analysis
- **Basic Statistics**: Mean, median, standard deviation, min, and max.
- **Advanced Statistics**: Quartile statistics, skewness, kurtosis, and normality tests.
- **Correlation Analysis**: Heatmap and strong correlations (|r| > 0.5).

### 🔍 Deep Analysis
- **Distribution Analysis**: Histograms and density plots.
- **Scatter Plots**: Visualize relationships between two numeric features.
- **Outlier Analysis**: Box plots for detecting outliers.

### 💡 AI Assistant
- Ask questions about your data and get detailed analysis.
- View the generated Python code used for the analysis.

### 📥 Download Full Analysis
- Download all analysis results, visualizations, and AI-generated insights in a ZIP file.

### Tools and Libraries Used

- **Streamlit**: For building the interactive web application.
- **Pandas**: For data manipulation and analysis.
- **SciPy**: For statistical computations and tests.
- **Plotly**: For creating interactive visualizations.
- **Pillow (PIL)**: For handling image files.
- **Zipfile**: For creating downloadable ZIP archives.
- **LLaMA 3.3**: (via Groq) For AI-powered data analysis and insights.
- **PandasAI**: For intelligent data analysis and automated insights using `SmartDataframe`.

### How to run?
Basic Usage (Without Downloading Analysis)                                                                                                       
If you do not want to download the analysis, run the app.py file using the following command:                                                      
streamlit run app.py                                                                                                                    

Advanced Usage (With Downloadable Reports)                                                                                                     
If you want to enable the feature to download the analysis report, run the app2.py file using the following command:                           
streamlit run app2.py
