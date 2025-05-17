import streamlit as st

st.set_page_config(page_title="Welcome to PDF-Q&A App", page_icon="🏠")

st.title("🚀 Welcome to the PDF-Q&A Application")

st.write(
    "This application helps you work with your PDF documents interactively.\n"
    "Use the sidebar to navigate between different sections:\n\n"
    "1. **Upload and Index PDF**: Upload your PDF files and index their content for fast retrieval.\n"
    "2. **Question and Answering**: Ask single-turn questions about your indexed documents.\n"
    "3. **Conversational Chat**: Engage in a multi-turn chat that maintains context across messages."
)
