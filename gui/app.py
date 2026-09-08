import os
import tempfile
import streamlit as st
from PIL import Image

# Import central controller and schemas
from main import NeckCADController
from core.schema import ImageStatus

# Page Configuration
st.set_page_config(
    page_title="NECK-CAD | Salivary Cytology Decision Support",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 20px; }
    .metric-card { background-color: #F3F4F6; padding: 15px; border-radius: 8px; border-left: 5px solid #1E3A8A; }
    .warning-card { background-color: #FEF3C7; padding: 15px; border-radius: 8px; border-left: 5px solid #D97706; }
    .danger-card { background-color: #FEE2E2; padding: 15px; border-radius: 8px; border-left: 5px solid #DC2626; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_controller():
    """Initialises and caches the NECK-CAD system controller."""
    return NeckCADController()


def main():
    controller = get_controller()

    # Sidebar Navigation & Session Setup
    st.sidebar.title("🔬 NECK-CAD System")
    st.sidebar.caption("Salivary Gland Cytopathology Decision Support")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Configuration")
    confidence_thresh = st.sidebar.slider(
        "Uncertainty Threshold", 0.50, 0.95, controller.config.CONFIDENCE_THRESHOLD, 0.05
    )
    controller.config.CONFIDENCE_THRESHOLD = confidence_thresh

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**UK Clinical Compliance Note**\n\n"
        "NECK-CAD is an assistive tool for computer-aided diagnosis. "
        "Final categorisation remains the responsibility of the reporting consultant pathologist."
    )

    # Main Header Block
    st.markdown('<div class="main-header">NECK-CAD Diagnostic Workbench</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Automated Fine Needle Aspiration (FNA) Quality Control, Categorisation, & Visualisation Engine</div>',
        unsafe_allow_html=True,
    )

    # File Ingestion Section
    st.subheader("1. Slide Upload & Field-of-View (FOV) Selection")
    uploaded_files = st.file_uploader(
        "Choose slide FOV images (PNG, JPEG, TIFF):",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Please upload one or more slide images to begin analysis.")
        return

    # Temporary File Handler for Session Ingestion
    temp_dir = tempfile.mkdtemp()
    file_paths = []

    for uploaded_file in uploaded_files:
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(temp_path)

    # Process Session Trigger
    if st.button("Run Clinical Analysis Pipeline", type="primary"):
        with st.spinner("Processing slide FOVs through quality gating, inference, and multi-image fusion..."):
            session = controller.create_session(file_paths)
            processed_session = controller.process_session(session)
            st.session_state["active_session"] = processed_session
        st.success("Analysis complete!")

    # Display Results if Active Session Exists
    if "active_session" in st.session_state:
        session = st.session_state["active_session"]
        res = session.aggregated_result

        st.markdown("---")
        st.subheader("2. Aggregated Diagnostic Synthesis")

        if res:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(label="Primary Categorisation", value=res.primary_category)
            with col2:
                st.metric(label="Milan System Category", value=res.milan_category)
            with col3:
                st.metric(label="Top Specific Diagnosis", value=res.specific_diagnosis)

            # Confidence Scores & Safeguards
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Confidence Metrics**")
                st.progress(res.confidence_scores.get("milan_confidence", 0.0), text=f"Milan Category Confidence: {res.confidence_scores.get('milan_confidence', 0.0):.1%}")
                st.progress(res.confidence_scores.get("entity_confidence", 0.0), text=f"Entity Confidence: {res.confidence_scores.get('entity_confidence', 0.0):.1%}")

            with col_b:
                st.write("**Safety & Audit Flags**")
                if res.is_uncertain:
                    st.markdown('<div class="warning-card">⚠️ <b>High Prediction Uncertainty Detected</b><br>Prediction confidence is below required operational thresholds. Manual microscopic review advised.</div>', unsafe_allow_html=True)
                elif res.is_ood:
                    st.markdown('<div class="danger-card">🚨 <b>Out-Of-Distribution (OOD) Warning</b><br>Morphological features deviate significantly from standard cytology training data.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="metric-card">✅ <b>Standard Confidence Bounds Passed</b><br>Nominal alignment with expected diagnostic bounds.</div>', unsafe_allow_html=True)

            # Differential Diagnoses
            st.write("**Differential Diagnosis Ranking**")
            if res.differential_diagnosis:
                diff_data = [{"Entity": ent, "Probability": f"{prob:.2%}"} for ent, prob in res.differential_diagnosis]
                st.table(diff_data)

        st.markdown("---")
        st.subheader("3. Field-of-View (FOV) Quality Gating & Visualisations")

        # Tabs for FOV Details
        fov_tabs = st.tabs([img.filename for img in session.images])

        for index, img_meta in enumerate(session.images):
            with fov_tabs[index]:
                col_img, col_cam, col_info = st.columns([1, 1, 1])

                with col_img:
                    st.write("**Original Image**")
                    if os.path.exists(img_meta.file_path):
                        raw_pil = Image.open(img_meta.file_path)
                        st.image(raw_pil, use_container_width=True)

                with col_cam:
                    st.write("**Grad-CAM Morphological Focus**")
                    if img_meta.image_id in session.explanations:
                        st.image(session.explanations[img_meta.image_id], use_container_width=True)
                    else:
                        st.caption("No heatmap visualisation generated for rejected image.")

                with col_info:
                    st.write("**Technical Quality Assessment**")
                    status_color = "green" if img_meta.status == ImageStatus.VALID else "red"
                    st.markdown(f"**Status:** <span style='color:{status_color}; font-weight:bold;'>{img_meta.status.value}</span>", unsafe_allow_html=True)
                    st.write(f"**Focus Sharpness Score:** {img_meta.quality_score:.2f}")

                    if img_meta.status != ImageStatus.VALID:
                        st.error(f"Rejection Reason: {img_meta.rejection_reason}")

        st.markdown("---")
        st.subheader("4. Clinical Documentation & Export")

        # Report Preview and Download
        report_text = controller.reporter.format_text_report(session)

        with st.expander("Preview Formatted Clinical Report"):
            st.code(report_text, language="text")

        st.download_button(
            label="📥 Download Clinical Report (.txt)",
            data=report_text,
            file_name=f"NECK_CAD_Report_{session.session_id}.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()