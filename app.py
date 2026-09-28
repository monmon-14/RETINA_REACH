"""
RetinaBot - Diabetic Retinopathy & Ophthalmic Technician AI Assistant
A clean, dedicated clinical chatbot powered by UiPath Robotic Process Automation (RPA).
Answers eye screening questions, ETDRS DR staging, OCT biomarkers, camera troubleshooting,
and dispatches automated hospital referrals.
"""

import streamlit as st
from chatbot_ui import render_technician_chatbot

# Page Configuration
st.set_page_config(
    page_title="RetinaBot | Ophthalmic Technician AI Co-Pilot",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<div class="main-header">👁️ RetinaBot: Diabetic Retinopathy Technician Co-Pilot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Specialized clinical decision support for ophthalmic screeners: '
        'answering questions on Diabetic Retinopathy grading, OCT biomarkers, imaging artifacts, and general eye conditions. '
        'Integrated with <b>UiPath Orchestrator</b> for one-click EMR referral automation.</div>',
        unsafe_allow_html=True
    )

    # Sidebar: Information & Hackathon Pitch Guide
    with st.sidebar:
        st.header("ℹ️ About RetinaBot")
        st.markdown("""
        **RetinaBot** is designed for ophthalmic technicians and mobile screening staff.
        
        **Core Clinical Knowledge:**
        - 🩺 **DR Severity (ETDRS)**: Stages 0 to 4 & the 4:2:1 Rule
        - 🔬 **OCT Biomarkers**: IRF, SRF, CST, Hyperreflective Foci, DRIL
        - 📷 **Camera Troubleshooting**: Fixing glare, blur, and small pupils
        - 🚨 **Red Flags**: Glaucoma, AMD, Cataract, CRVO, Detachment
        - ⏱️ **Triage Protocols**: Emergency (<24h), Urgent (1-4w), Routine
        
        ---
        **🤖 UiPath RPA Features:**
        - Auto-dispatches referrals to Orchestrator Queue (`DR_Specialist_Referrals`)
        - Real-time robot execution logs
        - Connects to Hospital EMR / Email alerts
        """)
        st.info("💡 **Tip**: Click any quick prompt chip to get instant clinical guidance.")

    # Render Chatbot & UiPath RPA Dispatch Center
    render_technician_chatbot()


if __name__ == "__main__":
    main()
