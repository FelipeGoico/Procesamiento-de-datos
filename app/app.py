import streamlit as st
import pandas as pd

st.set_page_config(page_title="Examen Final",
                   page_icon=":tada:", layout="wide")
st.title("Examen Procesamiento de Datos")
df = pd.read_csv("../data_processed.csv")

st.dataframe(df.head(30))
