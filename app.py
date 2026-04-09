import streamlit as st
import numpy as np
import cv2
import time
from PIL import Image

from streamlit_image_coordinates import streamlit_image_coordinates
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# Import pipeline modules
from src.seg.sam2_wrapper import SAM2BaseWrapper
from src.topo.depth_wrapper import DepthAnythingV2Wrapper
from src.curv.curvature import CurvatureAnalyzer
from src.match.engine import KnowledgeEngine
from src.topo.multiview_fuser import MultiViewFuser
from src.topo.calibration import ScaleCalibrator
from src.curv.curvature_pcd import PCDCurvatureAnalyzer

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
    .stProgress > div > div > div > div {
        background-color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# State Management
# ---------------------------------------------------------
if 'roi_prompts' not in st.session_state:
    st.session_state.roi_prompts = {} # {file_name: point}
if 'calib_points' not in st.session_state:
    st.session_state.calib_points = [] # [(x,y), (x,y)]
if 'calib_mm' not in st.session_state:
    st.session_state.calib_mm = 10.0
if 'masks_cache' not in st.session_state:
    st.session_state.masks_cache = {} # {file_name: {pt: (x,y), mask: np.array, viz: np.array}}

# ---------------------------------------------------------
# Caching Heavy Models
# ---------------------------------------------------------
# @st.cache_resource(show_spinner=False)
def load_models():
    """Load and cache the heavy AI models."""
    sam_wrapper = SAM2BaseWrapper(checkpoint_path="models/sam2/sam2_hiera_large.pt")
    depth_wrapper = DepthAnythingV2Wrapper(checkpoint_path="models/depth_anything_v2/depth_anything_v2_vitl.pth")
    curv_analyzer = CurvatureAnalyzer(smoothing_sigma=2.0)
    match_engine = KnowledgeEngine(db_path="data/database/film_properties.csv")
    fuser = MultiViewFuser(voxel_size=0.08)
    pcd_analyzer = PCDCurvatureAnalyzer(knn=40)
    calibrator = ScaleCalibrator()
    
    sam_wrapper.load_model()
    depth_wrapper.load_model()
    
    return sam_wrapper, depth_wrapper, curv_analyzer, match_engine, fuser, pcd_analyzer, calibrator

# ---------------------------------------------------------
# Translation Dictionary Extension
# ---------------------------------------------------------
text = {
    "title": {"en": "SG-TERRA: Topographic Evaluation & Recommendation", "ko": "SG-TERRA: 3D 표면 분석 및 필름 추천 시스템"},
    "desc": {"en": "Analyze steel plate 3D topography using single or multiple images with manual scale calibration for ultimate precision.", "ko": "단안 또는 다중 이미지와 실측 거리 보정을 통해 강판의 3D 지형을 정밀 분석하고 최적의 점착 필름을 추천받으세요."},
    "init": {"en": "Initializing AI Pipeline...", "ko": "AI 파이프라인(SAM 2, Depth-V2) 로드 중..."},
    "controls": {"en": "Controls & Settings", "ko": "제어 및 설정"},
    "upload": {"en": "Upload Plate Images", "ko": "강판 이미지 업로드 (복수 가능)"},
    "params": {"en": "Parameters", "ko": "파라미터 조정"},
    "sigma": {"en": "Gaussian Smoothing (\u03C3)", "ko": "가우시안 평활화 (\u03C3)"},
    "roughness": {"en": "Surface Roughness (Ra)", "ko": "목표 표면 조도 (Ra)"},
    "calib_title": {"en": "📏 Scale Calibration (Step 1)", "ko": "📏 1단계: 실측 거리 보정 (Calibration)"},
    "calib_desc": {"en": "Click TWO points on the first image and enter the real distance between them (mm).", "ko": "첫 번째 이미지에서 거리를 알고 있는 '두 지점'을 클릭하고 실측 거리(mm)를 입력하세요."},
    "calib_done": {"en": "Calibration Factor: 1px = {f:.4f}mm", "ko": "보정 완료: 1픽셀 ≈ {f:.4f}mm"},
    "roi_title": {"en": "🎯 Interactive ROI Selection (Step 2)", "ko": "🎯 2단계: 이미지별 분석 타겟(ROI) 지정"},
    "roi_desc": {"en": "Click on the target steel plate for EACH image below.", "ko": "아래 각 이미지에서 분석할 강판 영역(ROI)을 한 번씩 클릭해 주세요."},
    "btn_run": {"en": "Run Precise 3D Analysis 🚀", "ko": "초정밀 3D 통합 분석 실행 🚀"},
    "status_running": {"en": "Processing Sequential Pipeline...", "ko": "순차적 AI 파이프라인 가동 중..."},
    "status_done": {"en": "Analysis Complete!", "ko": "모든 분석 및 매칭이 완료되었습니다!"},
    "report_title": {"en": "📊 Final Topographic & Material Report", "ko": "📊 최종 표면 지형 및 점착제 매칭 리포트"},
    "metric_r": {"en": "Estimated Min Radius (R)", "ko": "추정 최소 곡률 반경 (R)"},
    "metric_curv": {"en": "Surface Variation (Avg)", "ko": "평균 표면 곡률 변동성"},
    "rec_top": {"en": "**🥇 Top Recommendation: {name} ({id})**", "ko": "**🥇 권장 최적 제품 (Top 1): {name} ({id})**"},
    "missing_image": {"en": "Please upload images to begin.", "ko": "사이드바에서 분석할 사진들을 먼저 업로드해 주세요."},
    "step_pcd": {"en": "Fusing Point Clouds & Analyzing Curvature...", "ko": "정합 퓨전 및 3D 점군 곡률 분석 중..."},
}

def t(key, **kwargs):
    s = text.get(key, {}).get(lang, key)
    return s.format(**kwargs) if kwargs else s

# ---------------------------------------------------------
# UI Core
# ---------------------------------------------------------
lang_toggle = st.sidebar.radio("🌐 Language", ["한국어", "English"])
lang = "ko" if lang_toggle == "한국어" else "en"

st.title(t("title"))
st.markdown(t("desc"))

with st.spinner(t("init")):
    sam, dw, ca, me, fuser, pa, calib = load_models()

st.sidebar.header(t("controls"))
uploaded_files = st.sidebar.file_uploader(t("upload"), type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

ca.sigma = st.sidebar.slider(t("sigma"), 0.5, 5.0, 2.0, 0.1)
roughness = st.sidebar.number_input(t("roughness"), value=1.0, step=0.1)
z_magnify = st.sidebar.slider("Depth Magnification (Z-Scale)", 0.1, 20.0, 1.0, 0.1)
aspect_corr = st.sidebar.slider("Aspect Correction (W/H)", 0.5, 2.0, 1.0, 0.05)
fov_corr = st.sidebar.slider("Camera FOV Scale", 0.5, 2.0, 1.0, 0.1)
surface_smooth = st.sidebar.slider("Surface Smoothing (Gaussian)", 0.0, 10.0, 2.0, 0.5)

# Logic for resizing UI and Processing images to prevent crashes/lag
MAX_UI_DIM = 1024
MAX_PROC_DIM = 2048 # Max dimension for AI models to ensure VRAM/Memory stability

def resize_for_proc(img_rgb):
    """Resize image for AI processing to a manageable size."""
    h, w = img_rgb.shape[:2]
    if max(h, w) <= MAX_PROC_DIM:
        return img_rgb, 1.0
    
    scale = MAX_PROC_DIM / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA), scale

def get_ui_img(img_rgb):
    """Resize image for UI display while keeping aspect ratio."""
    h, w = img_rgb.shape[:2]
    if max(h, w) <= MAX_UI_DIM:
        return img_rgb, 1.0
    
    scale = MAX_UI_DIM / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA), scale

def scale_point(pt, scale):
    """Scale UI coordinates back to original image size."""
    if pt is None: return None
    return (int(pt[0] / scale), int(pt[1] / scale))

def pcd_to_grid(pcd, target_h, target_w, scale_factor, aspect):
    """Project fused PCD to a 2D grid using its own bounding box."""
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return np.zeros((target_h, target_w))
    
    # Coordinates in mm
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Define grid based on PCD extent to avoid offset issues
    # But forced to target resolution for display
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    
    xi = np.linspace(min_x, max_x, target_w)
    yi = np.linspace(min_y, max_y, target_h)
    grid_x, grid_y = np.meshgrid(xi, yi)
    
    # Interpolate
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear', fill_value=np.min(z))
    
    # We also return the physical axes we derived
    return grid_z, xi, yi

if uploaded_files:
    # 1. Calibration (Always on the first image)
    st.subheader(t("calib_title"))
    st.info(t("calib_desc"))
    
    first_file = uploaded_files[0]
    first_bytes = np.asarray(bytearray(first_file.read()), dtype=np.uint8)
    first_file.seek(0) # Reset pointer
    img0_raw = cv2.cvtColor(cv2.imdecode(first_bytes, 1), cv2.COLOR_BGR2RGB)
    
    # Resize raw to processing size immediately
    img0_rgb, proc_s = resize_for_proc(img0_raw)
    
    # Resize for display
    img0_ui, s0 = get_ui_img(img0_rgb)
    
    c_val = streamlit_image_coordinates(img0_ui, key="calib_coord")
    if c_val:
        # Scale back to original
        new_pt = scale_point((c_val["x"], c_val["y"]), s0)
        if new_pt not in st.session_state.calib_points:
            st.session_state.calib_points.append(new_pt)
            if len(st.session_state.calib_points) > 2:
                st.session_state.calib_points = st.session_state.calib_points[-2:]

    if len(st.session_state.calib_points) == 2:
        p1, p2 = st.session_state.calib_points
        st.session_state.calib_mm = st.number_input("Real Distance (mm)", value=st.session_state.calib_mm, step=0.1)
        s_factor = calib.calibrate(p1, p2, st.session_state.calib_mm)
        st.success(t("calib_done", f=s_factor))
    else:
        st.warning(f"Currently selected: {len(st.session_state.calib_points)}/2 points.")

    # 2. ROI Selection for each image
    st.markdown("---")
    st.subheader(t("roi_title"))
    st.info("💡 **중요**: 모든 사진에서 **동일한 지점(예: 특정 모서리, 마크 등)**을 클릭해 주세요. 이 점이 3D 정합의 기준(Anchor)이 되어 훨씬 정확한 결과를 만듭니다.")
    
    num_files = len(uploaded_files)
    
    for idx, uf in enumerate(uploaded_files):
        with st.container():
            st.markdown(f"#### 📷 Image {idx+1}: {uf.name}")
            f_bytes = np.asarray(bytearray(uf.read()), dtype=np.uint8)
            uf.seek(0)
            img_raw = cv2.cvtColor(cv2.imdecode(f_bytes, 1), cv2.COLOR_BGR2RGB)
            
            # Resize raw to processing size
            img_rgb, s_proc = resize_for_proc(img_raw)
            
            # Resize for display
            img_ui, s_ui = get_ui_img(img_rgb)
            
            # Interactive Masking Logic
            display_img = img_ui.copy()
            if uf.name in st.session_state.roi_prompts:
                raw_pt = st.session_state.roi_prompts[uf.name]
                
                # Check cache
                cache = st.session_state.masks_cache.get(uf.name, {})
                if cache.get("pt") == raw_pt:
                    display_img = cache.get("viz")
                else:
                    # Run SAM 2 immediately
                    with st.spinner(f"Segmenting {uf.name}..."):
                        mask = sam.segment_target(img_rgb, prompt_points=np.array([[raw_pt[0], raw_pt[1]]]), prompt_labels=np.array([1]))
                        
                        # Create visual overlay for UI
                        # Downsample mask for viz
                        mask_ui = cv2.resize(mask.astype(np.uint8), (display_img.shape[1], display_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                        
                        # Use a cool blue translucent mask for chosen object
                        overlay = display_img.copy()
                        overlay[mask_ui == 1] = [45, 120, 255] # SG-TERRA Blue
                        display_img = cv2.addWeighted(display_img, 0.6, overlay, 0.4, 0)
                        
                        # Draw the prompt point
                        ui_pt = (int(raw_pt[0] * s_proc * s_ui), int(raw_pt[1] * s_proc * s_ui))
                        cv2.circle(display_img, ui_pt, 8, (255, 255, 255), -1)
                        cv2.circle(display_img, ui_pt, 6, (45, 120, 255), -1)
                        
                        st.session_state.masks_cache[uf.name] = {"pt": raw_pt, "mask": mask, "viz": display_img}

            # Show coordinates component with display_img (could be masked)
            roi_val = streamlit_image_coordinates(display_img, key=f"roi_{uf.name}")
            if roi_val:
                # Scale back
                raw_pt = scale_point((roi_val["x"], roi_val["y"]), s_ui)
                if st.session_state.roi_prompts.get(uf.name) != raw_pt:
                    st.session_state.roi_prompts[uf.name] = raw_pt
                    st.rerun() # Refresh to show new mask
            
            if uf.name in st.session_state.roi_prompts:
                st.write(f"✅ Selected Object Location: {st.session_state.roi_prompts[uf.name]}")
            else:
                st.error(f"Please Select Target for Image {idx+1}")
            
            st.markdown("---")

    # 3. Execution
    st.markdown("---")
    if st.button(t("btn_run"), type="primary"):
        if len(st.session_state.calib_points) < 2:
            st.error("Please complete Step 1 (Calibration) first.")
        elif len(st.session_state.roi_prompts) < num_files:
            st.error("Please select a target for ALL images in Step 2.")
        else:
            with st.status(t("status_running"), expanded=True) as status:
                rgb_list = []
                depth_list = []
                t_starts = time.time()
                
                progress_bar = st.progress(0)
                
                for i, uf in enumerate(uploaded_files):
                    st.write(f"▶️ Processing Image {i+1}/{num_files}: {uf.name}")
                    
                    f_bytes = np.asarray(bytearray(uf.read()), dtype=np.uint8)
                    uf.seek(0)
                    img_raw = cv2.cvtColor(cv2.imdecode(f_bytes, 1), cv2.COLOR_BGR2RGB)
                    img_rgb, _ = resize_for_proc(img_raw)
                    
                    # SAM 2
                    pt = st.session_state.roi_prompts[uf.name]
                    mask = sam.segment_target(img_rgb, prompt_points=np.array([[pt[0], pt[1]]]), prompt_labels=np.array([1]))
                    
                    # Depth
                    depth = dw.estimate_depth(img_rgb, mask=mask)
                    
                    rgb_list.append(img_rgb)
                    depth_list.append(depth)
                    progress_bar.progress((i + 1) / (num_files + 1))

                # Fusion & Precision Analysis
                st.write(f"▶️ {t('step_pcd')}")
                
                # Extract anchors for all images
                anchor_coords = [st.session_state.roi_prompts.get(uf.name) for uf in uploaded_files]
                
                # Pass z_magnify, fov_corr, and anchor_coords to fuser
                final_pcd = fuser.fuse_views(
                    rgb_list, depth_list, 
                    scale_factor=calib.scale_factor, 
                    z_scale=z_magnify,
                    fov_scale=fov_corr,
                    anchor_coords=anchor_coords
                )
                
                curvatures = pa.calculate_curvature(final_pcd)
                estimated_r = pa.estimate_min_radius(curvatures)
                avg_curv = np.mean(curvatures)
                
                t_total = time.time() - t_starts
                progress_bar.progress(1.0)
                status.update(label=t("status_done"), state="complete", expanded=False)

            # 4. Report
            st.subheader(t("report_title"))
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">R ≈ {estimated_r:.2f} mm</div><div class="metric-label">{t("metric_r")}</div></div>', unsafe_allow_html=True)
            with r2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_curv:.6f}</div><div class="metric-label">{t("metric_curv")}</div></div>', unsafe_allow_html=True)
            with r3:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{t_total:.2f} s</div><div class="metric-label">Processing Time</div></div>', unsafe_allow_html=True)

            # 2. Integrated 3D Topographic Analysis (Restored & Enhanced)
            st.markdown("---")
            st.subheader("🌐 Interactive 3D Topographic Surface (mm)")
            st.info("💡 마우스로 그래프를 회전하거나 확대하여 강판 지형을 입체적으로 분석할 수 있습니다.")
            
            # Prepare data from FUSED PCD (The Truth)
            h0, w0 = depth_list[0].shape
            
            with st.spinner("📦 Projecting Fused 3D Data to Grid..."):
                z_fused, x_coords, y_coords = pcd_to_grid(final_pcd, h0, w0, calib.scale_factor, aspect_corr)
            
            # Apply corrections to the derived axes for visual squaring
            x_plot_coords = x_coords * aspect_corr
            y_plot_coords = y_coords
            
            # Apply Gaussian Smoothing to remove high-frequency noise spritzes
            if surface_smooth > 0:
                with st.spinner("✨ Smoothing Surface..."):
                    z_fused = gaussian_filter(z_fused, sigma=surface_smooth)
            
            # Use magnification
            z_fused = z_fused * z_magnify
            
            # Simple downsampling for interaction
            ds = 4 if max(h0, w0) > 1000 else 2
            z_plot = z_fused[::ds, ::ds]
            x_plot = x_plot_coords[::ds]
            y_plot = y_plot_coords[::ds]

            fig_surface = go.Figure(data=[go.Surface(
                z=z_plot, x=x_plot, y=y_plot,
                colorscale='Viridis',
                contours = {
                    "z": {"show": True, "usecolormap": True, "project": {"z": True}}
                }
            )])
            
            fig_surface.update_layout(
                title="Fused Multi-View Surface Analysis (mm)",
                scene = dict(
                    xaxis_title='Width (mm)',
                    yaxis_title='Height (mm)',
                    zaxis_title='Depth (mm)',
                    aspectmode='data'
                ),
                height=800
            )
            st.plotly_chart(fig_surface, use_container_width=True)

            # Interactive 3D Fused Explorer (Raw PCD)
            with st.expander("🌍 Explore Fused 3D Point Cloud (Raw PCD)", expanded=False):
                pts = np.asarray(final_pcd.points)
                colors = np.asarray(final_pcd.colors)
                # Sample for performance if needed
                if len(pts) > 100000:
                    idx = np.random.choice(len(pts), 100000, replace=False)
                    pts, colors = pts[idx], colors[idx]
                
                fig = go.Figure(data=[go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode='markers', marker=dict(size=1.5, color=colors)
                )])
                fig.update_layout(height=700, margin=dict(l=0,r=0,b=0,t=0), scene=dict(aspectmode='data'))
                st.plotly_chart(fig, use_container_width=True)

            # Matching Engine
            st.markdown("---")
            recs = me.recommend(measured_curvature=estimated_r, measured_roughness=roughness)
            if recs:
                st.success(t("rec_success"))
                st.info(t("rec_top", name=recs[0]['film_name'], id=recs[0]['film_id']))
                st.dataframe(recs, use_container_width=True)
            else:
                st.error(t("rec_error"))

else:
    st.info(t("missing_image"))
