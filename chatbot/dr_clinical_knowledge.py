"""
dr_clinical_knowledge.py
========================
High-grade clinical knowledge engine tailored for Ophthalmic Technicians and Eye Care Screeners.
Provides comprehensive clinical guidance on:
- Diabetic Retinopathy (ETDRS stages: 0 to 4)
- Diabetic Macular Edema (DME) & CSME definitions
- OCT Biomarkers (CST, IRF, SRF, Hyperreflective Foci, DRIL, EZ disruption)
- Fundus Camera & OCT Acquisition Troubleshooting (haze, glare, dilation, blink)
- General Ophthalmic Conditions & Red Flags (Glaucoma, AMD, Cataract, CRVO/BRVO, Retinal Detachment)
- Clinical Referral Urgency & Triage Matrix
"""

import re
from typing import Dict, List, Optional, Tuple, Any

# ETDRS Diabetic Retinopathy Severity Grading
DR_STAGES = {
    0: {
        "title": "No Diabetic Retinopathy (ETDRS Level 10)",
        "icon": "🟢",
        "features": "Clear retinal background, crisp optic disc margins, normal foveal avascular zone (FAZ), no microaneurysms, hemorrhages, or exudates.",
        "oct_findings": "Normal foveal contour, intact ellipsoid zone (EZ), no cystic spaces or neurosensory detachment.",
        "urgency": "Routine Screening (12 months)",
        "technician_action": "Record baseline fundus photo and OCT. Instruct patient on strict glycemic control (HbA1c target <7%) and schedule annual re-screening.",
        "icd_10": "E11.9 / E10.9 (Diabetes without ophthalmic complications)"
    },
    1: {
        "title": "Mild Non-Proliferative Diabetic Retinopathy (NPDR - ETDRS Level 20)",
        "icon": "🟡",
        "features": "Microaneurysms only (isolated tiny red round dots, usually temporal to the fovea). No hard exudates, cotton wool spots, or venous beading.",
        "oct_findings": "Generally normal central retinal thickness; occasional isolated tiny hyperreflective foci or focal capillary dilation outside fovea.",
        "urgency": "Routine Screening (9 to 12 months)",
        "technician_action": "Ensure high-resolution macula-centered capture. Verify whether macula is threatened. Remind patient regarding blood pressure and lipid monitoring.",
        "icd_10": "E11.319 (Type 2 DM with mild nonproliferative DR without macular edema)"
    },
    2: {
        "title": "Moderate Non-Proliferative Diabetic Retinopathy (NPDR - ETDRS Level 35-47)",
        "icon": "🟠",
        "features": "More than just microaneurysms, but less than Severe NPDR. Dot and blot hemorrhages across 1-3 quadrants, hard exudates (lipid deposits), cotton wool spots (nerve fiber layer infarcts).",
        "oct_findings": "May exhibit early diabetic macular edema (DME), intraretinal cystoid spaces, or increased Central Subfield Thickness (CST > 250 µm).",
        "urgency": "Semi-Urgent (Follow-up in 3 to 6 months)",
        "technician_action": "Capture both 45° 7-standard field or ultra-widefield fundus photos and high-density OCT raster/radial scans through the fovea. Flag for ophthalmologist review.",
        "icd_10": "E11.329 (Type 2 DM with moderate nonproliferative DR without macular edema)"
    },
    3: {
        "title": "Severe Non-Proliferative Diabetic Retinopathy (NPDR - ETDRS Level 53 - The '4:2:1' Rule)",
        "icon": "🔴",
        "features": "Meets at least ONE criterion of the international 4:2:1 rule: \n  • 4 quadrants of severe intraretinal hemorrhages (>20 per quadrant)\n  • 2 quadrants of definite venous beading (sausage-like vein constriction/dilation)\n  • 1 quadrant of prominent IRMA (Intraretinal Microvascular Abnormalities, tortuous shunt vessels).",
        "oct_findings": "Frequently accompanies diffuse macular edema, neurosensory detachment, or severe DRIL (Disorganization of Retinal Inner Layers).",
        "urgency": "Urgent Referral to Retinal Specialist (Within 2 to 4 weeks; 50% risk of progression to PDR within 1 year)",
        "technician_action": "Highlight 4:2:1 features in report. Dispatch urgent referral ticket via hospital system. Advise patient against strenuous valsalva activities.",
        "icd_10": "E11.339 (Type 2 DM with severe nonproliferative DR without macular edema)"
    },
    4: {
        "title": "Proliferative Diabetic Retinopathy (PDR - ETDRS Level 61+)",
        "icon": "🚨",
        "features": "Neovascularization of the Disc (NVD), Neovascularization Elsewhere (NVE), Preretinal or Vitreous Hemorrhage, or Tractional Retinal Detachment (TRD). High risk of irreversible sudden visual loss.",
        "oct_findings": "Vitreoretinal traction, hyperreflective fibrovascular membranes pulling on the inner retina, profound macular disorganization, or subretinal fluid.",
        "urgency": "EMERGENCY / IMMEDIATE Referral (Within 24 to 48 hours for Panretinal Photocoagulation [PRP] / Anti-VEGF / Vitrectomy)",
        "technician_action": "Immediately notify attending retina specialist. Do not delay. Prepare fundus fluorescein angiography (FFA) or OCT-A if ordered.",
        "icd_10": "E11.359 (Type 2 DM with proliferative DR without macular edema)"
    }
}

# OCT Biomarkers & Features
OCT_BIOMARKERS = {
    "cst": {
        "name": "Central Subfield Thickness (CST)",
        "definition": "Average retinal thickness in the central 1 mm circular zone centered on the fovea (ETDRS grid).",
        "normal_range": "240 to 280 µm (varies slightly by OCT device: Cirrus, Spectralis, Triton).",
        "pathology": "CST > 300-320 µm indicates center-involving diabetic macular edema (CI-DME) requiring anti-VEGF therapy.",
        "tip": "Ensure accurate foveal centration; miscentered scans artificially inflate or deflate CST measurements."
    },
    "irf": {
        "name": "Intraretinal Fluid (IRF)",
        "definition": "Hyporeflective, round-to-oval fluid-filled cystoid spaces located within the inner or outer nuclear layers of the neurosensory retina.",
        "clinical_meaning": "Hallmark of breakdown of the blood-retinal barrier in active diabetic macular edema (DME). Highly responsive to anti-VEGF injections.",
        "tip": "Check for 'sponge-like' diffuse retinal thickening vs discrete focal cystoid cavities."
    },
    "srf": {
        "name": "Subretinal Fluid (SRF)",
        "definition": "Accumulation of fluid beneath the neurosensory retina, separating it from the underlying Retinal Pigment Epithelium (RPE). Appears as a hyporeflective dome or pocket.",
        "clinical_meaning": "Occurs in ~15-30% of diabetic macular edema eyes. Indicates chronic RPE pump overload or inflammatory components. Also rule out central serous chorioretinopathy or wet AMD.",
        "tip": "Inspect the integrity of the overlying photoreceptor layer (EZ line)."
    },
    "dril": {
        "name": "Disorganization of Retinal Inner Layers (DRIL)",
        "definition": "Loss of distinct boundaries between the Ganglion Cell Layer (GCL), Inner Plexiform Layer (IPL), Inner Nuclear Layer (INL), and Outer Plexiform Layer (OPL).",
        "clinical_meaning": "Strong biomarker for structural damage and poor visual acuity prognosis, even after edema resolves.",
        "tip": "Look at the central 1mm foveal zone. If layer borders cannot be demarcated, note 'DRIL present'."
    },
    "ez": {
        "name": "Ellipsoid Zone (EZ) Disruption",
        "definition": "Interruption or attenuation of the 2nd hyperreflective band on OCT representing photoreceptor inner segment/outer segment junction.",
        "clinical_meaning": "Directly correlates with functional visual impairment and photoreceptor death. Permanent visual recovery may be limited if EZ is disrupted.",
        "tip": "Verify signal strength (Q-score > 6/10) to ensure attenuation is not due to low optical signal."
    },
    "hyperreflective_foci": {
        "name": "Hyperreflective Foci (HF)",
        "definition": "Small (<30 µm), discrete, well-circumscribed hyperreflective dots scattered throughout retinal layers, without posterior shadowing.",
        "clinical_meaning": "Represent activated extravasated microglia and subclinical lipid/protein precursors to hard exudates. Signifies active neuroinflammation in DR.",
        "tip": "Differentiate from hard exudates which are larger (>30 µm) and produce strong optical shadowing underneath."
    }
}

# Image Acquisition & Quality Troubleshooting Guide for Technicians
TROUBLESHOOTING_GUIDE = {
    "blur": {
        "issue": "Blurry or Out-of-Focus Fundus Image",
        "causes": "Patient blink, refractive error (high myopia/hyperopia), uncompensated diopter setting, breathing motion.",
        "technician_fix": [
            "Adjust the camera's diopter compensation ring (-10D to +10D) to match patient's refractive error.",
            "Ask patient to blink once, then keep eye wide open and fixate on the internal target cross.",
            "Use the joystick to fine-tune retinal vessel crispness before pressing the shutter."
        ]
    },
    "glare": {
        "issue": "White Glare / Crescent Light Reflection (Halo)",
        "causes": "Camera is too close or off-axis relative to the cornea; reflections off corneal surface.",
        "technician_fix": [
            "Pull joystick gently back to verify working distance (usually 40-50 mm from cornea).",
            "Center the two infrared alignment dots precisely within the corneal entry ring before capturing.",
            "Ensure the room ambient lighting is dimmed to minimize external glare reflections."
        ]
    },
    "dark_or_underexposed": {
        "issue": "Dark, Underexposed or Vignetted Image",
        "causes": "Small/undilated pupil (<3.5mm), patient positioning too far back, dense nuclear cataract.",
        "technician_fix": [
            "Switch camera to 'Small Pupil' mode if available (narrows flash aperture).",
            "Allow patient 5 minutes in a darkened room for physiological dark adaptation, or request pharmacological dilation (Tropicamide 1%) per doctor protocol.",
            "Increase flash intensity (+1 or +2 EV steps) if significant cataract haze is present."
        ]
    },
    "shadowing_eyelash": {
        "issue": "Dark Lower or Upper Crescent / Eyelash Shadowing",
        "causes": "Ptosis, drooping upper eyelid, or long eyelashes obscuring the optical path.",
        "technician_fix": [
            "Instruct patient to open both eyes wide ('look surprised'). Opening the contralateral eye naturally raises both lids.",
            "If ptosis is severe, gently tape the upper eyelid to the brow using micropore surgical tape, avoiding direct globe pressure."
        ]
    },
    "oct_low_signal": {
        "issue": "Low OCT Signal Strength / Grainy B-Scan",
        "causes": "Dry ocular surface (tear film breakup), cataract/vitreous floater over scan axis, misaligned z-offset.",
        "technician_fix": [
            "Instill 1 drop of preservative-free artificial tears to re-establish smooth optical corneal surface.",
            "Use auto-Z and auto-focus functions; tilt the scan beam slightly away from dense central cataract opacities.",
            "Ensure patient holds fixation for the 2-3 seconds needed for frame averaging."
        ]
    }
}

# General Eye Conditions & Red Flags
GENERAL_EYE_CONDITIONS = {
    "glaucoma": {
        "name": "Glaucoma (Optic Neuropathy)",
        "features": "Elevated Cup-to-Disc ratio (>0.6 or asymmetry >0.2 between eyes), thinning of Neuroretinal Rim (violation of ISNT rule), retinal nerve fiber layer (RNFL) wedge defects, disc hemorrhages.",
        "technician_tip": "Always examine the optic nerve head carefully. If cup is large or asymmetric, perform RNFL OCT scan and Tonometry (IOP)."
    },
    "amd": {
        "name": "Age-Related Macular Degeneration (AMD)",
        "features": "Drusen (yellow sub-RPE deposits under macula), RPE pigment mottling (Dry AMD). Subretinal fluid, choroidal neovascular membrane (CNVM), or macular hemorrhage (Wet AMD).",
        "technician_tip": "Differentiate drusen from hard exudates: Drusen are soft/cuticular mounds beneath the RPE on OCT; hard exudates are intraretinal lipid deposits with posterior shadowing."
    },
    "cataract": {
        "name": "Cataract (Lens Opacity)",
        "features": "Diffuse yellow/brown haze obscuring retinal view, generalized loss of fundus contrast and sharpness.",
        "technician_tip": "Note media opacity in screening logs. If fundus cannot be graded due to cataract, mark 'Ungradable - Media Opacity' and refer for cataract evaluation."
    },
    "crvo_brvo": {
        "name": "Retinal Vein Occlusion (CRVO / BRVO)",
        "features": "'Blood and thunder' fundus appearance: extensive flame-shaped and blot hemorrhages along venous distribution, marked tortuosity, disc edema, cotton wool spots.",
        "technician_tip": "Urgent referral required. Check blood pressure immediately (frequently associated with severe systemic hypertension)."
    },
    "retinal_detachment": {
        "name": "Retinal Detachment / Tear (Emergency Red Flag)",
        "features": "Symptoms of sudden shower of floaters, flashes of light (photopsia), and dark curtain over vision. Visualized as corrugated, mobile, folded retina with loss of choroidal pattern.",
        "technician_tip": "🚨 IMMEDIATE EMERGENCY. Do not dilate if acute angle closure is suspected. Patient must see vitreoretinal surgeon within 24 hours."
    }
}


class ClinicalEyeAssistant:
    """
    Intelligent clinical assistant engine for Ophthalmic Technicians.
    Evaluates queries, provides structured clinical explanations, parses symptoms,
    and formats actionable advice for fundus & OCT workflows.
    """

    def __init__(self):
        self.dr_stages = DR_STAGES
        self.oct_biomarkers = OCT_BIOMARKERS
        self.troubleshooting = TROUBLESHOOTING_GUIDE
        self.general_conditions = GENERAL_EYE_CONDITIONS

    def get_stage_info(self, stage_idx: int) -> Optional[Dict[str, Any]]:
        """Returns details for a specific ETDRS DR stage (0-4)."""
        return self.dr_stages.get(stage_idx)

    def answer_query(self, user_text: str) -> str:
        """
        Processes a natural language question from an ophthalmic technician
        and generates a high-quality, structured clinical response.
        """
        text = user_text.lower().strip()

        # 1. 4:2:1 Rule / Severe NPDR
        if "4:2:1" in text or "421" in text or "severe npdr" in text or "severe non-proliferative" in text:
            stage3 = self.dr_stages[3]
            return (
                f"### {stage3['icon']} Severe NPDR & The International '4:2:1' Rule\n\n"
                f"The **4:2:1 Rule** is the gold-standard ETDRS classification criteria defining **Severe Non-Proliferative Diabetic Retinopathy (NPDR)**. "
                f"An eye is diagnosed with Severe NPDR if it meets **at least ONE** of the following:\n\n"
                f"1. **`4` Quadrants of Severe Intraretinal Hemorrhages**: >20 distinct dot/blot hemorrhages in each of the 4 quadrants.\n"
                f"2. **`2` Quadrants of Definite Venous Beading**: Segmental tortuosity and sausage-like focal caliber variations in retinal veins.\n"
                f"3. **`1` Quadrant of Prominent IRMA**: Intraretinal Microvascular Abnormalities (abnormal shunt vessels that do NOT cross over major blood vessels).\n\n"
                f"**Clinical Significance & Risk:**\n"
                f"- Patients with Severe NPDR have a **50% probability** of advancing to Proliferative DR (PDR) within 12 months if untreated!\n\n"
                f"**Technician Action Protocol:**\n"
                f"- **Referral Urgency**: {stage3['urgency']}.\n"
                f"- Request widefield imaging or 7-standard field photos to verify peripheral non-perfusion.\n"
                f"- Check OCT for center-involving macular edema."
            )

        # 2. PDR / Proliferative
        if any(w in text for w in ["pdr", "proliferative", "neovascularization", "nvd", "nve", "vitreous hemorrhage"]):
            stage4 = self.dr_stages[4]
            return (
                f"### {stage4['icon']} Proliferative Diabetic Retinopathy (PDR)\n\n"
                f"**Pathophysiology & Defining Features:**\n"
                f"Proliferative DR occurs when widespread retinal capillary non-perfusion causes chronic ischemia, triggering massive release of Vascular Endothelial Growth Factor (**VEGF**). "
                f"This induces fragile, abnormal new blood vessels:\n"
                f"- **NVD**: Neovascularization at or within 1 disc diameter of the Optic Disc.\n"
                f"- **NVE**: Neovascularization Elsewhere along vascular arcades.\n"
                f"- **Vitreous / Preretinal Hemorrhage**: Fragile vessels rupture, bleeding into the retrohyaloid space or vitreous gel.\n"
                f"- **Tractional Retinal Detachment (TRD)**: Fibrovascular scaffolds contract, pulling retina away from RPE.\n\n"
                f"**Referral Timeline:**\n"
                f"- **{stage4['urgency']}**\n\n"
                f"**Technician Workflow:**\n"
                f"- Immediately contact the on-call retina specialist.\n"
                f"- Prepare the patient for possible anti-VEGF injection, Panretinal Photocoagulation (PRP laser), or Pars Plana Vitrectomy (PPV)."
            )

        # 3. Mild or Moderate NPDR
        if "mild" in text and ("npdr" in text or "diabetic" in text or "retinopathy" in text):
            s = self.dr_stages[1]
            return f"### {s['icon']} {s['title']}\n\n**Visual Hallmarks:** {s['features']}\n\n**OCT Findings:** {s['oct_findings']}\n\n**Urgency:** {s['urgency']}\n\n**Technician Note:** {s['technician_action']}"

        if "moderate" in text and ("npdr" in text or "diabetic" in text or "retinopathy" in text):
            s = self.dr_stages[2]
            return f"### {s['icon']} {s['title']}\n\n**Visual Hallmarks:** {s['features']}\n\n**OCT Findings:** {s['oct_findings']}\n\n**Urgency:** {s['urgency']}\n\n**Technician Note:** {s['technician_action']}"

        # 4. Diabetic Macular Edema (DME / CSME)
        if any(w in text for w in ["dme", "macular edema", "csme", "edema"]):
            return (
                "### 💧 Diabetic Macular Edema (DME) & CSME Guidelines\n\n"
                "Diabetic Macular Edema is the **leading cause of moderate-to-severe visual acuity loss** in patients with diabetes.\n\n"
                "**ETDRS Clinically Significant Macular Edema (CSME) Criteria (Biomicroscopy):**\n"
                "1. Thickening of retina located at or within **500 µm** (1/3 disc diameter) of foveal center.\n"
                "2. Hard exudates at or within **500 µm** of foveal center if associated with adjacent retinal thickening.\n"
                "3. Retinal thickening of **1 disc diameter (1500 µm)** or larger, any part of which is within 1 disc diameter of foveal center.\n\n"
                "**OCT Device Interpretation:**\n"
                "- Look at the **Central Subfield Thickness (CST)**: Normal is ~240-280 µm. CST > 300 µm indicates Center-Involving DME (CI-DME).\n"
                "- Identify fluid distribution: **Intraretinal Fluid (IRF)** in nuclear layers vs **Subretinal Fluid (SRF)** under neurosensory retina."
            )

        # 5. OCT Fluid Questions (IRF vs SRF vs DRIL)
        if any(w in text for w in ["irf", "srf", "intraretinal fluid", "subretinal fluid", "cst", "dril", "oct", "biomarker"]):
            return self._explain_oct_biomarkers(text)

        # 6. Camera / Acquisition Troubleshooting
        if any(w in text for w in ["blur", "glare", "dark", "underexposed", "eyelash", "reflection", "camera", "quality", "troubleshoot", "artifact", "focus"]):
            return self._troubleshoot_image_acquisition(text)

        # 7. General Eye Conditions (Glaucoma, AMD, Cataract, Detachment, Vein Occlusion)
        if any(w in text for w in ["glaucoma", "amd", "macular degeneration", "drusen", "cataract", "crvo", "brvo", "vein occlusion", "detachment", "flashes", "floaters"]):
            return self._explain_general_eye_conditions(text)

        # 8. Referral / Triage Urgency & Emergency Red Flags
        if any(w in text for w in ["emergency", "red flag", "danger", "referral", "triage", "urgency", "timeline", "when to refer", "schedule"]):
            return (
                "### 🚨 Ophthalmic Emergency Red Flags (<24-Hour Immediate Referral)\n\n"
                "The following conditions require **immediate, same-day referral to a vitreoretinal specialist / eye emergency unit**:\n\n"
                "1. **Retinal Detachment / Tear**: Sudden shower of floaters, flashes of light (photopsia), or a dark curtain progressing across visual field.\n"
                "2. **Proliferative DR (PDR) with Vitreous Hemorrhage**: Sudden massive, painless loss of vision with dense retrohyaloid/vitreous blood obscuring retinal details.\n"
                "3. **Central Retinal Artery Occlusion (CRAO)**: Sudden catastrophic painless vision loss; visualizes 'cherry-red spot' at fovea with pale retina.\n"
                "4. **Acute Angle-Closure Glaucoma**: Severe ocular pain, frontal headache, nausea/vomiting, colored halos around lights, mid-dilated fixed pupil, and rock-hard globe (IOP >40-60 mmHg).\n\n"
                + self._get_triage_matrix()
            )

        # 9. General Retinopathy Overview / Stages Summary
        if any(w in text for w in ["stage", "classification", "etdrs", "what is dr", "diabetic retinopathy"]):
            return self._get_all_stages_summary()

        # Default fallback with helpful technician guide options
        return (
            "👋 **Ophthalmic Technician AI Assistant Ready.**\n\n"
            "I can assist you with clinical interpretations, imaging troubleshooting, and UiPath hospital automation. Here are some common topics you can ask me:\n\n"
            "• **DR Staging**: *'Explain the 4:2:1 rule for Severe NPDR'*, *'What are hallmarks of PDR?'*\n"
            "• **OCT Biomarkers**: *'What is the difference between IRF and SRF?'*, *'How is CST interpreted?'*\n"
            "• **Camera Troubleshooting**: *'How do I fix corneal glare?'*, *'Patient has small pupils and dark photos'*\n"
            "• **Other Eye Diseases**: *'How to tell drusen from hard exudates?'*, *'Glaucoma cup-to-disc signs'*\n"
            "• **Triage & Referral**: *'What is the referral urgency timeline for each stage?'*\n"
            "• **UiPath Automation**: Click the **⚡ Trigger UiPath Referral** button on the right to auto-dispatch patient referral to the hospital EMR queue!"
        )

    def _explain_oct_biomarkers(self, text: str) -> str:
        """Explains OCT biomarkers requested by the user."""
        res = ["### 🔬 OCT Biomarker Clinical Reference Guide\n"]

        if "irf" in text or "intraretinal" in text or "fluid" in text:
            b = self.oct_biomarkers["irf"]
            res.append(f"**1. {b['name']} ({b['definition']})**\n- **Clinical Meaning:** {b['clinical_meaning']}\n- **Technician Tip:** {b['tip']}\n")

        if "srf" in text or "subretinal" in text or "fluid" in text:
            b = self.oct_biomarkers["srf"]
            res.append(f"**2. {b['name']} ({b['definition']})**\n- **Clinical Meaning:** {b['clinical_meaning']}\n- **Technician Tip:** {b['tip']}\n")

        if "cst" in text or "thickness" in text:
            b = self.oct_biomarkers["cst"]
            res.append(f"**3. {b['name']}**\n- **Normal Range:** {b['normal_range']}\n- **Pathology Threshold:** {b['pathology']}\n- **Technician Tip:** {b['tip']}\n")

        if "dril" in text or "layer" in text:
            b = self.oct_biomarkers["dril"]
            res.append(f"**4. {b['name']}**\n- **Definition:** {b['definition']}\n- **Clinical Meaning:** {b['clinical_meaning']}\n- **Technician Tip:** {b['tip']}\n")

        if len(res) == 1:  # none matched specifically, provide summary
            for key, b in self.oct_biomarkers.items():
                res.append(f"• **{b['name']}**: {b['clinical_meaning']}")

        return "\n".join(res)

    def _troubleshoot_image_acquisition(self, text: str) -> str:
        """Provides camera acquisition and artifact fixing advice."""
        res = ["### 📷 Fundus & OCT Image Quality Troubleshooting\n"]

        matched = False
        if any(w in text for w in ["blur", "sharp", "focus", "clear"]):
            b = self.troubleshooting["blur"]
            res.append(f"**Issue: {b['issue']}**\n*Causes:* {b['causes']}\n*Corrective Actions:*\n" + "\n".join(f"- {f}" for f in b['technician_fix']) + "\n")
            matched = True

        if any(w in text for w in ["glare", "halo", "reflection", "white"]):
            b = self.troubleshooting["glare"]
            res.append(f"**Issue: {b['issue']}**\n*Causes:* {b['causes']}\n*Corrective Actions:*\n" + "\n".join(f"- {f}" for f in b['technician_fix']) + "\n")
            matched = True

        if any(w in text for w in ["dark", "underexposed", "pupil", "dilation"]):
            b = self.troubleshooting["dark_or_underexposed"]
            res.append(f"**Issue: {b['issue']}**\n*Causes:* {b['causes']}\n*Corrective Actions:*\n" + "\n".join(f"- {f}" for f in b['technician_fix']) + "\n")
            matched = True

        if any(w in text for w in ["eyelash", "shadow", "lid", "ptosis"]):
            b = self.troubleshooting["shadowing_eyelash"]
            res.append(f"**Issue: {b['issue']}**\n*Causes:* {b['causes']}\n*Corrective Actions:*\n" + "\n".join(f"- {f}" for f in b['technician_fix']) + "\n")
            matched = True

        if not matched:
            for key, b in self.troubleshooting.items():
                res.append(f"• **{b['issue']}**: {b['causes']} ➔ *{b['technician_fix'][0]}*")

        return "\n".join(res)

    def _explain_general_eye_conditions(self, text: str) -> str:
        """Explains general ophthalmic conditions and differential diagnosis."""
        res = ["### 🩺 General Ophthalmic Pathology Differential\n"]

        if "glaucoma" in text:
            g = self.general_conditions["glaucoma"]
            res.append(f"**{g['name']}**\n- **Hallmark Features:** {g['features']}\n- **Technician Note:** {g['technician_tip']}\n")

        if any(w in text for w in ["amd", "macular degeneration", "drusen"]):
            a = self.general_conditions["amd"]
            res.append(f"**{a['name']}**\n- **Hallmark Features:** {a['features']}\n- **Technician Note:** {a['technician_tip']}\n")

        if "cataract" in text:
            c = self.general_conditions["cataract"]
            res.append(f"**{c['name']}**\n- **Hallmark Features:** {c['features']}\n- **Technician Note:** {c['technician_tip']}\n")

        if any(w in text for w in ["crvo", "brvo", "vein occlusion"]):
            v = self.general_conditions["crvo_brvo"]
            res.append(f"**{v['name']}**\n- **Hallmark Features:** {v['features']}\n- **Technician Note:** {v['technician_tip']}\n")

        if any(w in text for w in ["detachment", "flashes", "floaters"]):
            d = self.general_conditions["retinal_detachment"]
            res.append(f"**{d['name']}**\n- **Hallmark Features:** {d['features']}\n- **Technician Note:** {d['technician_tip']}\n")

        return "\n".join(res)

    def _get_triage_matrix(self) -> str:
        """Returns referral urgency protocol."""
        return (
            "### ⏱️ Ophthalmic Triage & Specialist Referral Protocol\n\n"
            "| Condition / Stage | Urgency Window | Clinical Action |\n"
            "|---|---|---|\n"
            "| **PDR / Vitreous Hemorrhage / Retinal Detachment** | 🚨 **Immediate (<24-48 Hours)** | Direct on-call retina surgeon referral for laser/vitrectomy |\n"
            "| **Severe NPDR (4:2:1 Rule) or Center-Involving DME** | 🔴 **Urgent (1 to 4 Weeks)** | Specialist evaluation for Anti-VEGF / Focal laser |\n"
            "| **Moderate NPDR (Non-center DME)** | 🟠 **Semi-Urgent (3 to 6 Months)** | Repeat fundus photography & OCT raster scans |\n"
            "| **Mild NPDR (isolated microaneurysms)** | 🟡 **Routine (9 to 12 Months)** | Primary care glycemic optimization (HbA1c < 7%) |\n"
            "| **No DR (Healthy Retina)** | 🟢 **Annual (12 Months)** | Standard annual diabetic eye screening |"
        )

    def _get_all_stages_summary(self) -> str:
        """Returns summary of all 5 ETDRS stages."""
        lines = ["### 📋 International Clinical Diabetic Retinopathy Severity Scale (ETDRS)\n"]
        for idx in sorted(self.dr_stages.keys()):
            s = self.dr_stages[idx]
            lines.append(f"**{s['icon']} Grade {idx}: {s['title']}**\n- **Findings:** {s['features']}\n- **Referral:** *{s['urgency']}*\n")
        return "\n".join(lines)
