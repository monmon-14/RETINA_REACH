"""
standalone_app.py
=================
Standalone launcher for the Ophthalmic Technician Clinical AI Assistant & UiPath RPA Dispatcher.
Run directly via:
    streamlit run chatbot/standalone_app.py
"""

import streamlit as st

try:
    from .chatbot_ui import render_technician_chatbot
except ImportError:
    from chatbot_ui import render_technician_chatbot

st.set_page_config(
    page_title="RetinaBot | Technician Co-Pilot",
    page_icon="👁️",
    layout="wide"
)

render_technician_chatbot()
