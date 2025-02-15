# utils/data_loading.py

import pandas as pd
import streamlit as st
from config import SAMPLE_DATASETS

def load_data(uploaded_file, sample_data):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    elif sample_data != "None":
        return pd.DataFrame(SAMPLE_DATASETS[sample_data])
    return None