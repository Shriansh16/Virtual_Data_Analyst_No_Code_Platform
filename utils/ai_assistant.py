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

