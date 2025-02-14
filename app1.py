import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_groq import ChatGroq
from pandasai import SmartDataframe
import os
from PIL import Image  # For handling image files

# Initialize LLM model
model = ChatGroq(
    groq_api_key="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",
    model="llama-3.3-70b-versatile",
    temperature=0.5
)

st.title("🔍 Virtual Data Analyst - No code Platform")

# File Upload
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=['csv'])

if uploaded_file is not None:
    # Read CSV file
    data = pd.read_csv(uploaded_file)
    st.subheader("📌 Preview of Uploaded Data")
    st.write(data.head(3))

    # Convert to SmartDataframe
    df = SmartDataframe(data, config={"llm": model})

    # **Automatically Generate Insights**
    st.subheader("📊 Insights")

    # 1. **Basic Summary Statistics**
    st.write("**🔹 Summary Statistics:**")
    st.write(data.describe())

    # 2. **Missing Values**
    missing_values = data.isnull().sum()
    if missing_values.any():
        st.write("**⚠️ Missing Values:**")
        st.write(missing_values[missing_values > 0])
    else:
        st.success("✅ No missing values found.")

    # 3. **AI-Generated Trends & Anomalies**
    st.write("**📈 AI-Generated Insights:**")
    insights_prompt = """
    Provide deeper insights about this dataset, including:
    - Unusual patterns or anomalies
    - Interesting correlations
    - Data segments with the most variance
    - Surprising trends that might not be obvious
    """
    with st.spinner("🔎 Analyzing dataset..."):
        insights = df.chat(insights_prompt)
        st.write(insights)

    # **Graphs for Visual Insights**
    st.subheader("📊 Interesting Graphs")

    # 1. **Correlation Heatmap**
    st.write("**🔹 Correlation Heatmap**")
    numeric_data = data.select_dtypes(include=["number"])  # Select only numeric columns

    if not numeric_data.empty:
      fig, ax = plt.subplots(figsize=(10, 6))
      sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
      st.pyplot(fig)
    else:
      st.warning("⚠️ No numeric data available for correlation analysis.")

    # 2. **Feature Distribution (Histogram)**
    st.write("**🔹 Feature Distribution**")
    selected_feature = st.selectbox("Choose a feature to visualize:", data.columns)
    
    if selected_feature:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data[selected_feature].dropna(), kde=True, bins=30, color="blue")
        ax.set_title(f"Distribution of {selected_feature}")
        st.pyplot(fig)

    # 3. **Boxplot to Detect Outliers**
    st.write("**🔹 Boxplot to Detect Outliers**")
    selected_feature_box = st.selectbox("Choose a feature for outlier detection:", data.columns)
    
    if selected_feature_box:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=data[selected_feature_box], color="red")
        ax.set_title(f"Boxplot of {selected_feature_box}")
        st.pyplot(fig)

    # **User Input for Custom Analysis**
    st.subheader("💡 Ask Your Own Question")
    prompt = st.text_area("Type your query:")

    if st.button("Generate"):
        if prompt:
            with st.spinner("🔄 Generating response..."):
                response = df.chat(prompt)
                
                # Check if the response is a file path to an image
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