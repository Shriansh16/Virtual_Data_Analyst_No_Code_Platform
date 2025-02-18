import os
import zipfile
import io

# Function to create a downloadable ZIP file
def create_zip_file(data, analysis_results, visualizations, ai_analysis):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        # Save data as CSV
        zip_file.writestr("data.csv", data.to_csv(index=False))

        # Save analysis results
        for name, df in analysis_results.items():
            zip_file.writestr(f"{name}.csv", df.to_csv(index=False))

        # Save visualizations as images
        for name, fig in visualizations.items():
            fig.write_image(f"{name}.png")
            zip_file.write(f"{name}.png")

        # Save AI analysis and code
        zip_file.writestr("ai_analysis.txt", ai_analysis)

        # Create a summary report
        summary = f"""
        # Analysis Summary Report

        ## Data Overview
        - Rows: {data.shape[0]}
        - Columns: {data.shape[1]}
        - Missing Values: {data.isnull().sum().sum()}
        - Duplicate Rows: {data.duplicated().sum()}

        ## Statistical Analysis
        - Basic Statistics: See basic_stats.csv
        - Quartile Statistics: See quartile_stats.csv
        - Correlation Matrix: See correlation_matrix.csv

        ## AI Analysis
        {ai_analysis}
        """
        zip_file.writestr("summary_report.md", summary)

    zip_buffer.seek(0)
    return zip_buffer