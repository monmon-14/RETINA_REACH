"""
chatbot_ui.py
=============
Interactive, technician-centric Chatbot UI component for Diabetic Retinopathy
and General Ophthalmic Screening.
Pure clinical question-answering and UiPath robotic process automation.
"""

import streamlit as st
from typing import Dict, Any, Optional
from dr_clinical_knowledge import ClinicalEyeAssistant
from uipath_bridge import UiPathBridge


def init_chat_session():
    """Initializes session state variables for chat history and UiPath client."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "👋 **Hello! I am your Ophthalmic Technician Co-Pilot.**\n\n"
                    "I am here to assist you with eye screening questions, clinical protocols, and automated hospital workflows:\n\n"
                    "• **Diabetic Retinopathy (DR)**: ETDRS grading stages (0 to 4), the international 4:2:1 rule for Severe NPDR, and PDR risks.\n"
                    "• **Diabetic Macular Edema (DME)**: CSME criteria and Central Subfield Thickness (CST) guidelines.\n"
                    "• **OCT Biomarkers**: Intraretinal Fluid (IRF), Subretinal Fluid (SRF), Hyperreflective Foci (HF), and DRIL.\n"
                    "• **Image Acquisition Troubleshooting**: Correcting corneal glare, blur, eyelid shadowing, and small pupil underexposure.\n"
                    "• **General Eye Conditions**: Glaucoma (cup-to-disc ratio), AMD (drusen vs hard exudates), Cataracts, and Retinal Vein Occlusion (CRVO/BRVO).\n"
                    "• **UiPath Automation**: Auto-dispatching urgent referrals and clinical summaries to the hospital EMR queue.\n\n"
                    "*Type your question below or click any quick topic to get started!*"
                )
            }
        ]

    if "assistant_engine" not in st.session_state:
        st.session_state.assistant_engine = ClinicalEyeAssistant()

    if "uipath_bridge" not in st.session_state:
        st.session_state.uipath_bridge = UiPathBridge()


def render_technician_chatbot():
    """
    Renders the technician chatbot and UiPath RPA action center.
    """
    init_chat_session()
    assistant = st.session_state.assistant_engine
    uipath = st.session_state.uipath_bridge

    # Header section
    st.markdown("### 💬 Ophthalmic Technician AI Assistant & UiPath Automation")
    st.caption("Clinical decision support, camera & OCT troubleshooting, and UiPath robotic process automation for eye screening.")

    # Layout into two columns: Left for Chat / Q&A, Right for UiPath RPA Action Center
    chat_col, rpa_col = st.columns([3, 2], gap="large")

    with chat_col:
        st.markdown("#### 🧑‍⚕️ Clinical Co-Pilot")

        # Quick Action Buttons
        st.markdown("**Quick Topics for Technicians:**")
        q_col1, q_col2, q_col3 = st.columns(3)

        user_prompt_from_button = None

        with q_col1:
            if st.button("🔴 Severe NPDR (4:2:1)", use_container_width=True):
                user_prompt_from_button = "Explain the 4:2:1 rule for Severe NPDR"
            if st.button("💧 OCT: IRF vs SRF", use_container_width=True):
                user_prompt_from_button = "What is the difference between IRF and SRF on OCT?"

        with q_col2:
            if st.button("📷 Fix Image Glare/Blur", use_container_width=True):
                user_prompt_from_button = "How do I fix corneal glare and blur on fundus photos?"
            if st.button("⏱️ Triage Referral Matrix", use_container_width=True):
                user_prompt_from_button = "What are the referral urgency timelines for each DR stage?"

        with q_col3:
            if st.button("👁️ Drusen vs Exudates", use_container_width=True):
                user_prompt_from_button = "How to differentiate drusen in AMD from hard exudates in DR?"
            if st.button("🚨 Emergency Eye Red Flags", use_container_width=True):
                user_prompt_from_button = "What are the emergency eye red flags requiring immediate referral?"

        # Container for chat messages
        chat_container = st.container(height=450)
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Chat Input
        user_input = st.chat_input("Ask a clinical, OCT, camera artifact, or referral question...")

        final_prompt = user_prompt_from_button or user_input

        if final_prompt:
            # Append user message
            st.session_state.chat_history.append({"role": "user", "content": final_prompt})
            
            # Generate assistant response
            reply = assistant.answer_query(final_prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

    with rpa_col:
        st.markdown("#### ⚡ UiPath RPA Dispatch Center")
        st.info("Directly bridge clinical findings to hospital workflows. Dispatch high-priority referrals and sync EMR without manual paperwork.")

        with st.form("uipath_referral_form"):
            patient_id = st.text_input("Patient ID / Medical Record No. (MRN)", value="MRN-2026-8412")
            
            detected_cond = st.selectbox(
                "Condition to Flag",
                [
                    "Severe NPDR (4:2:1 Rule)",
                    "Proliferative DR (PDR)",
                    "Diabetic Macular Edema (DME)",
                    "Moderate NPDR",
                    "Mild NPDR",
                    "No DR (Healthy Retina)",
                    "Other Ophthalmic Anomaly (Glaucoma/AMD/CRVO)"
                ],
                index=0
            )

            urgency_options = [
                "🚨 EMERGENCY Consult (<24-48 Hours)",
                "Urgent Specialist Consult (2-4 Weeks)",
                "Follow-up in 3-6 Months",
                "Routine Screening (12 Months)"
            ]
            default_urg_idx = 0 if ("Proliferative" in detected_cond or "EMERGENCY" in detected_cond) else (1 if "Severe" in detected_cond else 2)
            urgency_choice = st.selectbox("Triage Urgency Window", urgency_options, index=default_urg_idx)

            conf_val = st.slider("Diagnostic Confidence (%)", min_value=50, max_value=100, value=93, step=1)
            notes = st.text_area("Technician Clinical Observations", value="Suspected center-involving macular edema and extensive intraretinal hemorrhages. Requesting retinal specialist evaluation.")
            specialist_email = st.text_input("Specialist / Department Email", value="retina.oncall@sih-hospital.org")

            submit_rpa = st.form_submit_button("🚀 Trigger UiPath RPA Robot", use_container_width=True)

        if submit_rpa:
            with st.spinner("🤖 UiPath Robot Initializing Orchestrator Queue Job..."):
                tx_result = uipath.dispatch_referral_to_uipath(
                    patient_id=patient_id,
                    predicted_condition=detected_cond,
                    confidence_pct=float(conf_val),
                    urgency=urgency_choice,
                    technician_notes=notes,
                    doctor_email=specialist_email
                )

            st.success(f"✅ UiPath Robot Job Completed: `{tx_result['transaction_id']}`")

            # UiPath Execution Telemetry
            with st.expander("📊 UiPath Robot Execution Telemetry & Logs", expanded=True):
                st.markdown(f"**Runtime Mode:** `{tx_result['mode']}`")
                st.markdown(f"**Target Queue:** `{tx_result['queue_name']}` | **Status:** `{tx_result['status']}`")
                
                # Render terminal-like logs
                log_text = "\n".join(tx_result["logs"])
                st.code(log_text, language="bash")

            with st.expander("📄 View UiPath Orchestrator JSON Payload"):
                st.json(tx_result["payload"])

        # History of dispatched transactions
        history = uipath.get_history()
        if history:
            with st.expander(f"📜 Session RPA Queue History ({len(history)} tickets)"):
                for item in history:
                    st.write(f"• **{item['timestamp']}** | `{item['transaction_id']}` | Patient: `{item['patient_mrn']}` | Status: `{item['status']}`")
