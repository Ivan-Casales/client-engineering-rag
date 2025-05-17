import os
from dotenv import load_dotenv
import streamlit as st
import requests
from typing import List, Dict

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.title("Conversational Chat")
st.write("This chat keeps context across multiple turns.")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Your message:", key="chat_input")
    submit = st.form_submit_button("Send")
    if submit and user_input:
        try:
            resp = requests.post(
                f"{API_BASE_URL}/api/chat",
                json={
                    "message": user_input,
                    "history": st.session_state["chat_history"]
                },
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            st.session_state["chat_history"] = data["history"]
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")

if st.session_state["chat_history"]:
    for turn in st.session_state["chat_history"]:
        st.markdown(f"**You:** {turn['user']}")
        st.markdown(f"**Assistant:** {turn['assistant']}")
else:
    st.info("No messages yet. Start the conversation by typing above.")