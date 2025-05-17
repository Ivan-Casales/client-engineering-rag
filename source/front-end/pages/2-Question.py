import os
from dotenv import load_dotenv
import streamlit as st
import requests
from typing import Optional

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.title("Question and Answering")

st.write(
    "Use this interface to ask questions about your documents."
)

st.header("Ask a Question")
question: Optional[str] = st.text_input("Your question:")
if st.button("Ask"):
    if not question:
        st.warning("Please enter a question before submitting.")
    else:
        try:
            endpoint = f"{API_BASE_URL}/api/ask"
            response = requests.post(
                endpoint,
                json={"question": question},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer")
                st.markdown("**Answer:**")
                st.write(answer)
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")
