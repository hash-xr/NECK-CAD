# gui/app.py
import os
import tempfile
import sys
from PIL import Image
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

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

# High-End Dark Cyber-Medical Styling
st.markdown(
    """
    <style>
    /* Global Background and Typography Overrides */
    @import url('https://googleapis.com');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #0B0F19;
        color: #E2E8F0;
    }
    
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1F2937;
    }
    
    /* Elegant Glowing Gradient Header */
    .main-header { 
        font-size: 2.6rem; 
        font-weight: 800; 
        background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        letter-spacing: -0.05em;
    }
    
    .sub-header { 
        font-size: 1.1rem; 
        color: #9CA3AF; 
        margin-bottom: 30px; 
        font-weight: 400;
    }
    
    /* Modernised Dashboard Cards */
    .metric-card { 
        background: #1F2937; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #374151;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-top: 4px solid #10B981;
    }
    .warning-card { 
        background: #2A2415; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #78350F;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-top: 4px solid #F59E0B;
    }
    .danger-card { 
        background: #2D1A1A; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #991B1B;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-top: 4px solid #EF4444;
    }
    
    /* Custom Stylised Metric Blocks for the Aggregated View */
    .cyber-metric {
        background: #111827;
        border: 1px solid #1F2937;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: inset 0 2px 4px 0 rgba(0,0,0,0.06);
    }
    .cyber-metric-label {
        font-size: 0.85rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .cyber-metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #60A5FA;
    }
    
    /* Section Separation Headers */
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #F3F4F6;
        margin-top: 25px;
        margin-bottom: 15px;
        border-bottom: 1px solid #1F2937;
        padding-bottom: 8px;
    }
    
    /* Clean up native tables to look dark and flat */
    .stTable table {
        background-color: #111827 !important;
        color: #E2E8F0 !important;
        border-collapse: collapse;
        border-radius: 8px;
        overflow: hidden;
    }
    .stTable th {
        background-color: #1F2937 !important;
        color: #9CA3AF !important;
        text-transform: uppercase;
        font-size: 0.8rem;
    }
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
    st.sidebar.title("🔬 NECK-CAD")
    st.sidebar.caption("Salivary Gland Cytopathology Pipeline")
    st.sidebar.markdown("---")

    st.sidebar.subheader("🎛️ Pipeline Thresholds")
    confidence_thresh = st.sidebar.slider(
        "Uncertainty Gating Threshold", 0.50, 0.95, controller.config.CONFIDENCE_THRESHOLD, 0.05
    )
    controller.config.CONFIDENCE_THRESHOLD = confidence_thresh

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size:0.85rem; color:#9CA3AF; background:#1F2937; padding:12px; border-radius:8px; border:1px solid #374151;'>"
        "<b>🇬🇧 UK Clinical Compliance Note</b><br><br>"
        "NECK-CAD is configured as an algorithmic decision support system. "
        "Final interpretive validation remains the strict domain of the reporting consultant pathologist."
        "</div>",
        unsafe_allow_html=True
    )

    # Main Header Block
    st.markdown('<div class="main-header">NECK-CAD Diagnostic Workbench</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Automated Fine Needle Aspiration (FNA) Quality Control, Categorisation, & Visualisation Engine</div>',
        unsafe_allow_html=True,
    )

    # File Ingestion Section
    st.markdown('<div class="section-title">1. Slide Upload & Field-of-View (FOV) Selection</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Choose slide FOV images (PNG, JPEG, TIFF):",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if not uploaded_files:
        st.info("Please upload one or more high-resolution slide images to begin cellular aggregation.")
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
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Run Clinical Analysis Pipeline", type="primary", use_container_width=True):
        with st.spinner("Processing slide FOVs through quality gating, inference, and multi-image fusion..."):
            session = controller.create_session(file_paths)
            processed_session = controller.process_session(session)
            st.session_state["active_session"] = processed_session
        st.success("Analysis complete!")

    # Display Results if Active Session Exists
    if "active_session" in st.session_state:
        session = st.session_state["active_session"]
        res = session.aggregated_result

        st.markdown('<div class="section-title">2. Aggregated Diagnostic Synthesis</div>', unsafe_allow_html=True)

        if res:
            # Custom styled cyber-medical metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"<div class='cyber-metric'><div class='cyber-metric-label'>Primary Categorisation</div><div class='cyber-metric-value'>{res.primary_category}</div></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='cyber-metric'><div class='cyber-metric-label'>Milan System Category</div><div class='cyber-metric-value'>{res.milan_category}</div></div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div class='cyber-metric'><div class='cyber-metric-label'>Top Specific Diagnosis</div><div class='cyber-metric-value'>{res.specific_diagnosis}</div></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Confidence Scores & Safeguards
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**📊 Mathematical Confidence Vectors**")
                st.progress(res.confidence_scores.get("milan_confidence", 0.0), text=f"Milan Category Confidence: {res.confidence_scores.get('milan_confidence', 0.0):.1%}")
                st.progress(res.confidence_scores.get("entity_confidence", 0.0), text=f"Entity Selection Confidence: {res.confidence_scores.get('entity_confidence', 0.0):.1%}")

            with col_b:
                st.write("**🛡️ Safety Gating & Audit Flags**")
                if res.is_uncertain:
                    st.markdown('<div class="warning-card">⚠️ <b>High Prediction Uncertainty Detected</b><br>Mathematical confidence metrics fall below required operational bounds. Manual microscopic validation required.</div>', unsafe_allow_html=True)
                elif res.is_ood:
                    st.markdown('<div class="danger-card">🚨 <b>Out-Of-Distribution (OOD) Warning</b><br>Morphological features deviate significantly from known cytology training distributions. Check for artifacts.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="metric-card">✅ <b>Standard Confidence Bounds Passed</b><br>Nominal alignment verified across expected diagnostic vector fields.</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.write("**📋 Differential Diagnosis Ranking**")
            if res.differential_diagnosis:
                diff_data = [{"Entity Code / Category": ent, "Probability Weight": f"{prob:.2%}"} for ent, prob in res.differential_diagnosis]
                st.table(diff_data)

        st.markdown('<div class="section-title">3. Field-of-View (FOV) Quality Gating & Visualisations</div>', unsafe_allow_html=True)

                # Tabs for FOV Details
        fov_tabs = st.tabs([f"🖼️ {img.filename}" for img in session.images])

        for index, img_meta in enumerate(session.images):
            with fov_tabs[index]:
                col_img, col_cam, col_info = st.columns([1, 1, 1])

                with col_img:
                    st.write("**Original Microscopic Input**")
                    if os.path.exists(img_meta.file_path):
                        raw_pil = Image.open(img_meta.file_path)
                        st.image(raw_pil, use_container_width=True)

                with col_cam:
                    st.write("**Grad-CAM Attention Heatmap**")
                    if img_meta.image_id in session.explanations:
                        st.image(session.explanations[img_meta.image_id], use_container_width=True)
                    else:
                        st.caption("No heatmap generated. Slide input was rejected at quality gating layer.")

                with col_info:
                    st.write("**Technical Quality Scorecard**")
                    status_color = "#10B981" if img_meta.status == ImageStatus.VALID else "#EF4444"
                    st.markdown(f"**Gating Assessment Status:** <span style='color:{status_color}; font-weight:bold;'>{img_meta.status.value}</span>", unsafe_allow_html=True)
                    st.write(f"**Calculated Sharpness (Laplacian Variance):** {img_meta.quality_score:.2f}")

                    if img_meta.status != ImageStatus.VALID:
                        st.error(f"Pipeline Bypass Triggered: {img_meta.rejection_reason}")

        st.markdown('<div class="section-title">4. Clinical Documentation & Export</div>', unsafe_allow_html=True)

        # Report Preview and Download
        report_text = controller.reporter.format_text_report(session)

        with st.expander("👁️ Preview Generated Clinical Audit Transcript"):
            st.code(report_text, language="text")

        st.download_button(
            label="📥 Export Report to Electronic Health Record (.txt)",
            data=report_text,
            file_name=f"NECK_CAD_Report_{session.session_id}.txt",
            mime="text/plain",
            use_container_width=True
        )


if __name__ == "__main__":
    main()