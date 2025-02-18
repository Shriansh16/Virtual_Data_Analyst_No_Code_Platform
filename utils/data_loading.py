import pandas as pd
import streamlit as st
import mysql.connector
import time
from config import SAMPLE_DATASETS

# Define MySQL connection parameters
DB_CONFIG = {
    "host": "localhost",
    "user": "shriansh",  
    "password": "shriansh99",
    "database": "datasets"
}

def connect_to_db():
    """Create and return a MySQL connection."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as e:
        st.error(f"Database connection error: {e}")
        return None

def generate_unique_table_name():
    """Generates a unique table name using a timestamp."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    return f"uploaded_data_{timestamp}"

def upload_to_mysql(data, table_name):
    """Uploads a DataFrame to MySQL, handling missing values and using parameterized queries."""
    conn = connect_to_db()
    if conn is None:
        return

    cursor = conn.cursor()

    # Create table dynamically based on DataFrame columns
    columns = ", ".join([f"`{col}` TEXT" for col in data.columns])
    create_table_query = f"CREATE TABLE IF NOT EXISTS `{table_name}` ({columns});"
    cursor.execute(create_table_query)

    # Prepare parameterized query
    placeholders = ", ".join(["%s"] * len(data.columns))
    insert_query = f"INSERT INTO `{table_name}` VALUES ({placeholders})"

    # Insert data using executemany()
    rows = [tuple(None if pd.isna(v) else v for v in row) for _, row in data.iterrows()]
    cursor.executemany(insert_query, rows)

    conn.commit()
    cursor.close()
    conn.close()


def load_data(uploaded_file, sample_data):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format")
            return None
        table_name = generate_unique_table_name()  # Generate unique table name
        upload_to_mysql(data, table_name)  # Upload to MySQL
        return data

    elif sample_data != "None":
        return pd.DataFrame(SAMPLE_DATASETS.get(sample_data, {}))

    return None
