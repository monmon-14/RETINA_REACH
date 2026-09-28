"""
test_chatbot.py
===============
Automated verification tests for the Ophthalmic Technician Clinical Assistant
and UiPath RPA Dispatcher.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dr_clinical_knowledge import ClinicalEyeAssistant
from uipath_bridge import UiPathBridge


def test_clinical_assistant():
    print("Testing ClinicalEyeAssistant...")
    assistant = ClinicalEyeAssistant()

    # Test 1: 4:2:1 Rule query
    res_421 = assistant.answer_query("Can you explain the 4:2:1 rule?")
    assert "4:2:1" in res_421 or "Severe NPDR" in res_421, "Failed 4:2:1 rule check"
    print("  [PASS] 4:2:1 Rule Query verified.")

    # Test 2: PDR & Neovascularization
    res_pdr = assistant.answer_query("What is PDR and neovascularization?")
    assert "Proliferative" in res_pdr, "Failed PDR query check"
    print("  [PASS] PDR Query verified.")

    # Test 3: OCT Fluid (IRF vs SRF)
    res_oct = assistant.answer_query("What is the difference between IRF and SRF on OCT?")
    assert "Intraretinal Fluid" in res_oct or "IRF" in res_oct, "Failed OCT query check"
    print("  [PASS] OCT Biomarkers Query verified.")

    # Test 4: Camera Troubleshooting (glare)
    res_glare = assistant.answer_query("How to fix glare and reflection?")
    assert "Glare" in res_glare or "working distance" in res_glare.lower(), "Failed Glare troubleshooting check"
    print("  [PASS] Camera Glare Troubleshooting verified.")

    # Test 5: General Eye (Glaucoma)
    res_glaucoma = assistant.answer_query("How does glaucoma look like?")
    assert "Glaucoma" in res_glaucoma or "Cup-to-Disc" in res_glaucoma, "Failed Glaucoma check"
    print("  [PASS] General Eye Disease (Glaucoma) verified.")

    # Test 6: Emergency Red Flags / Triage
    res_emergency = assistant.answer_query("What are the emergency eye red flags?")
    assert "EMERGENCY" in res_emergency or "Detachment" in res_emergency, "Failed Emergency red flags check"
    print("  [PASS] Emergency Red Flags & Triage verified.")


def test_uipath_bridge():
    print("\nTesting UiPathBridge...")
    bridge = UiPathBridge(force_demo_mode=True)

    result = bridge.dispatch_referral_to_uipath(
        patient_id="PT-TEST-001",
        predicted_condition="Proliferative DR (PDR)",
        confidence_pct=95.4,
        urgency="🚨 EMERGENCY Consult (<24-48 Hours)",
        technician_notes="Preretinal hemorrhage and neovascularization on superior arcade.",
        oct_biomarkers=["Intraretinal Fluid", "CST > 340um"],
        doctor_email="retina_urgent@hospital.org"
    )

    assert result["success"] is True, "Dispatch failed"
    assert "transaction_id" in result, "Missing transaction ID"
    assert result["transaction_id"].startswith("UIPATH-TX-"), f"Invalid tx format: {result['transaction_id']}"
    assert len(result["logs"]) > 0, "Missing robot logs"
    assert result["payload"]["itemData"]["Priority"] == "High", "Priority should be High for emergency"

    print(f"  [PASS] Dispatched mock UiPath transaction: {result['transaction_id']}")
    print(f"  [PASS] UiPath Queue item priority: {result['payload']['itemData']['Priority']}")
    print(f"  [PASS] Robot telemetry generated ({len(result['logs'])} log lines).")

    history = bridge.get_history()
    assert len(history) == 1, "History count mismatch"
    print("  [PASS] Transaction history logging verified.")


if __name__ == "__main__":
    test_clinical_assistant()
    test_uipath_bridge()
    print("\nALL AUTOMATED TESTS PASSED SUCCESSFULLY! (100% GREEN)")
