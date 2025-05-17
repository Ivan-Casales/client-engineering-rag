import os
from dotenv import load_dotenv
import streamlit as st
import requests
from typing import Optional

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.title("📄 PDF UPLOADER")

st.write(
    "Use this interface to upload a PDF for indexing."
)

st.header("Upload and Index PDF")
uploaded_file = st.file_uploader("Upload a PDF to index", type=["pdf"])
if uploaded_file is not None:
    if st.button("Index PDF"):
        try:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf"
                )
            }
            endpoint = f"{API_BASE_URL}/api/upload-pdf"
            response = requests.post(endpoint, files=files, timeout=30)
            if response.status_code == 200:
                detail = response.json().get("detail", "")
                st.success(f"Success: {detail}")
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")