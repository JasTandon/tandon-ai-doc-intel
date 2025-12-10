import streamlit as st
import sys
import os

print("--- DEBUG APP STARTING ---")
print(f"Python: {sys.version}")
print(f"CWD: {os.getcwd()}")

st.title("Debug Mode")
st.write("If you can see this, Streamlit itself is working.")

try:
    import plotly
    st.write(f"Plotly version: {plotly.__version__}")
except ImportError as e:
    st.error(f"Failed to import plotly: {e}")

try:
    import tandon_ai_doc_intel
    st.write(f"Library found: {tandon_ai_doc_intel.__file__}")
except ImportError as e:
    st.error(f"Failed to import library: {e}")
    st.write(f"Sys.path: {sys.path}")

st.write("End of debug script.")

