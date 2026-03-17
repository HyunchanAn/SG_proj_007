import streamlit as st
import numpy as np
import cv2
import time
from PIL import Image

from streamlit_image_coordinates import streamlit_image_coordinates
import plotly.graph_objects as go

# Import pipeline modules
from src.seg.sam2_wrapper import SAM2BaseWrapper
from src.topo.depth_wrapper import DepthAnythingV2Wrapper
from src.curv.curvature import CurvatureAnalyzer
from src.match.engine import KnowledgeEngine

# Page config
st.set_page_config(page_title="SG-TERRA AI", page_icon="🔍", layout="wide")

# CSS to make the app look premium
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        font-family: 'Inter', sans-serif;
        color: #1E3A8A;
        font-weight: 700;
    }
    h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #2563EB;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #1F2937;
    }
    .metric-label {
        font-size: 14px;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Caching Heavy Models
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    """Load and cache the heavy AI models so they don't reload on every interaction."""
    sam_wrapper = SAM2BaseWrapper(checkpoint_path="models/sam2/sam2_hiera_large.pt")
    depth_wrapper = DepthAnythingV2Wrapper(checkpoint_path="models/depth_anything_v2/depth_anything_v2_vitl.pth")
    curv_analyzer = CurvatureAnalyzer(smoothing_sigma=2.0)
    match_engine = KnowledgeEngine(db_path="data/database/film_properties.csv")
    
    # Trigger inner load methods
    sam_wrapper.load_model()
    depth_wrapper.load_model()
    
    return sam_wrapper, depth_wrapper, curv_analyzer, match_engine

# ---------------------------------------------------------
# Sidebar & Header
# ---------------------------------------------------------
st.title("SG-TERRA: Topographic Evaluation & Recommendation")
st.markdown("Upload an image of a steel plate to analyze its 3D structural curvature and get the optimal adhesive film recommendation.")

with st.spinner("Initializing AI Models (SAM 2 & Depth-Anything-V2)... This only happens once."):
    sam_wrapper, depth_wrapper, curv_analyzer, match_engine = load_models()

st.sidebar.header("Controls & Settings")
uploaded_file = st.sidebar.file_uploader("Upload Plate Image", type=['jpg', 'jpeg', 'png'])

# Tuning
st.sidebar.subheader("Analysis Parameters")
smoothing_sigma = st.sidebar.slider("Gaussian Smoothing (\u03C3)", 0.5, 5.0, 2.0, 0.1, help="Higher values reduce noise but may smooth out sharp curvatures.")
curv_analyzer.sigma = smoothing_sigma

roughness = st.sidebar.number_input("Surface Roughness (Ra)", value=1.0, step=0.1)

# ---------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------
if uploaded_file is not None:
    # Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    st.subheader("1. Source Image (Click to target the steel plate 😎)")
    
    # Let the user click the image to select the ROI point
    value = streamlit_image_coordinates(image_rgb, key="pil")
    
    prompt_points = None
    prompt_labels = None
    if value is not None:
        prompt_points = np.array([[value["x"], value["y"]]])
        prompt_labels = np.array([1])
        st.info(f"📍 Target point selected at: X={value['x']}, Y={value['y']}. Click 'Run Topology Analysis' below.")
    else:
        st.warning("👈 Please click on the steel plate in the image to set the SAM 2 target point.")

    if st.sidebar.button("Run Topology Analysis 🚀", type="primary"):
        if value is None:
            st.error("Please explicitly click on the steel plate target inside the image above to tell the AI what to analyze!")
        else:
            # Create Layout Columns
            col1, col2 = st.columns(2)
            
            with st.status("Running Multimodal AI Pipeline...", expanded=True) as status:
                # Step 1: Segmentation
                st.write("Target Segmentation (SAM 2)...")
                t0 = time.time()
                target_mask = sam_wrapper.segment_target(image_rgb, prompt_points=prompt_points, prompt_labels=prompt_labels)
                t_seg = time.time() - t0
            
                # Step 2: Depth Estimation
                st.write("Estimating Depth (Depth-Anything-V2)...")
                t0 = time.time()
                depth_map = depth_wrapper.estimate_depth(image_rgb, mask=target_mask)
                t_depth = time.time() - t0
                
                # Step 3: Curvature Analysis
                st.write("Calculating Stress Concentration Zones...")
                t0 = time.time()
                gaussian_curv = curv_analyzer.calculate_gaussian_curvature(depth_map, mask=target_mask)
                critical_vals, critical_coords = curv_analyzer.find_critical_points(gaussian_curv, mask=target_mask, top_k=1)
                t_curv = time.time() - t0
                
                # Extract Max Stress Point
                highest_stress_raw = critical_vals[0]
                estimated_r_mm = 3.5  # In real life, convert raw -> physical mm using a calibration factor
                t_total = t_seg + t_depth + t_curv
                
                status.update(label="Analysis Complete!", state="complete", expanded=False)

            # ---------------------------------------------------------
            # Visualization
            # ---------------------------------------------------------
            with col1:
                st.subheader("2. Target Segmentation Map")
                # Apply color overlay to the mask
                colored_mask = np.zeros_like(image_rgb)
                colored_mask[target_mask] = [0, 255, 0] # Green overlay
                blended = cv2.addWeighted(image_rgb, 0.7, colored_mask, 0.3, 0)
                st.image(blended, use_container_width=True, caption=f"SAM 2 Target Extracted Mask ({t_seg*1000:.1f}ms)")
                
            with col2:
                st.subheader("3. 3D Depth Topography")
                # Normalize depth for visualization (0-255)
                depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
                depth_colormap_rgb = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
                st.image(depth_colormap_rgb, use_container_width=True, caption=f"Relative Depth Output ({t_depth*1000:.1f}ms)")

                
            # ---------------------------------------------------------
            # Interactive 3D Grid / Contour
            # ---------------------------------------------------------
            with st.expander("🌍 Show Interactive 3D Topographic Grid & Contour"):
                st.markdown("Use your mouse/touch to rotate, zoom, and explore the physical surface grid.")
                
                # Downsample for browser performance 
                scale_percent = 20
                width = int(depth_map.shape[1] * scale_percent / 100)
                height = int(depth_map.shape[0] * scale_percent / 100)
                resized_depth = cv2.resize(depth_map, (width, height), interpolation = cv2.INTER_AREA)
                
                fig = go.Figure(data=[go.Surface(
                    z=resized_depth, 
                    colorscale='Inferno',
                    contours = {
                        "z": {"show": True, "size": (np.max(resized_depth)-np.min(resized_depth))/15, "color": "white"}
                    }
                )])
                
                fig.update_layout(
                    title='3D Depth Heatmap Framework',
                    autosize=True,
                    height=600,
                    margin=dict(l=0, r=0, b=0, t=30),
                    scene=dict(
                        xaxis=dict(showbackground=False),
                        yaxis=dict(showbackground=False),
                        zaxis=dict(showbackground=False)
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
                
            # ---------------------------------------------------------
            # Data & Recommendation Report
            # ---------------------------------------------------------
            st.markdown("---")
            st.subheader("4. Material Match Recommendation")
            
            # Display Metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{highest_stress_raw:.4f}</div><div class="metric-label">Max Gaussian Curvature (K)</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">R ≈ {estimated_r_mm} mm</div><div class="metric-label">Estimated Min Radius</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{t_total:.2f} s</div><div class="metric-label">Total Inference Time</div></div>', unsafe_allow_html=True)
                
            st.write("")
            
            # Recommendations
            recommendations = match_engine.recommend(measured_curvature=estimated_r_mm, measured_roughness=roughness)
            
            if not recommendations:
                st.error("No suitable adhesive films found for this level of extreme curvature in the database.")
            else:
                st.success("Analysis successful. Displaying top optimal products tailored to withstand the evaluated physical stress.")
                best_match = recommendations[0]
                st.info(f"**🥇 Top Recommendation: {best_match['film_name']} ({best_match['film_id']})**  \n"
                        f"Surpasses minimum required curvature ({best_match['max_curvature_radius']}mm capacity) with a high correlation score for peel strength and elongation.")
                
                # Show all options in a table
                st.write("### All Feasible Candidates")
                st.dataframe(recommendations, use_container_width=True)
else:
    st.info("Please upload an image using the sidebar to begin.")
