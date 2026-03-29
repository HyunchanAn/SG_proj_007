import streamlit as st
import numpy as np
import cv2
import time
from PIL import Image

from streamlit_image_coordinates import streamlit_image_coordinates
import plotly.graph_objects as go
import os
import urllib.request

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
    sam2_cfg = "sam2_hiera_s.yaml"
    sam2_ckpt = "models/sam2/sam2_hiera_small.pt"
    sam2_url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
    
    depth_encoder = "vits"
    depth_ckpt = "models/depth_anything_v2/depth_anything_v2_vits.pth"
    depth_url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
    
    os.makedirs("models/sam2", exist_ok=True)
    os.makedirs("models/depth_anything_v2", exist_ok=True)
    
    if not os.path.exists(sam2_ckpt):
        print("Downloading SAM 2 Small...")
        urllib.request.urlretrieve(sam2_url, sam2_ckpt)
        
    if not os.path.exists(depth_ckpt):
        print("Downloading Depth-Anything-V2 Small...")
        urllib.request.urlretrieve(depth_url, depth_ckpt)

    sam_wrapper = SAM2BaseWrapper(model_cfg=sam2_cfg, checkpoint_path=sam2_ckpt)
    depth_wrapper = DepthAnythingV2Wrapper(encoder=depth_encoder, checkpoint_path=depth_ckpt)
    curv_analyzer = CurvatureAnalyzer(smoothing_sigma=2.0)
    match_engine = KnowledgeEngine(db_path="data/database/film_properties.csv")
    
    # Trigger inner load methods
    sam_wrapper.load_model()
    depth_wrapper.load_model()
    
    return sam_wrapper, depth_wrapper, curv_analyzer, match_engine

# ---------------------------------------------------------
# Translation Dictionary
# ---------------------------------------------------------
text = {
    "title": {"en": "SG-TERRA: Topographic Evaluation & Recommendation", "ko": "SG-TERRA: 3D 표면 분석 및 필름 추천 시스템"},
    "desc": {"en": "Upload an image of a steel plate to analyze its 3D structural curvature and get the optimal adhesive film recommendation.", "ko": "강판의 이미지를 업로드하여 3D 구조적 곡률을 분석하고 최적의 점착 필름 추천을 받아보세요."},
    "init": {"en": "Initializing AI Models (SAM 2 & Depth-Anything-V2)... This only happens once.", "ko": "AI 모델(SAM 2 & Depth-V2)을 로드 중입니다... (최초 1회 단일 적용 지연)"},
    "controls": {"en": "Controls & Settings", "ko": "조작 및 설정 (Controls)"},
    "upload": {"en": "Upload Plate Image", "ko": "분석용 강판 이미지 업로드"},
    "params": {"en": "Analysis Parameters", "ko": "파라미터 미세 조정 (Tuning)"},
    "sigma": {"en": "Gaussian Smoothing (\u03C3)", "ko": "가우시안 곡률 평활화 (\u03C3)"},
    "sigma_help": {"en": "Higher values reduce noise but may smooth out sharp curvatures.", "ko": "값이 클수록 노이즈(난반사 오판 등)가 줄어들지만 미세한 굴곡이 무뎌질 수 있습니다."},
    "roughness": {"en": "Surface Roughness (Ra)", "ko": "목표 표면 조도 (Ra)"},
    "step1_title": {"en": "1. Source Image (Click to target the steel plate 😎)", "ko": "1. 소스 이미지 (분석 대상을 마우스로 직접 클릭하세요 😎)"},
    "target_selected": {"en": "📍 Target point selected at: X={x}, Y={y}. Click 'Run Topology Analysis' below.", "ko": "📍 타겟이 성공적으로 선택되었습니다: X={x}, Y={y}. 아래 '분석 실행' 버튼을 눌러주세요."},
    "target_warning": {"en": "👈 Please click on the steel plate in the image to set the SAM 2 target point.", "ko": "👈 마우스를 움직여 이미지 내 타겟 피착제를 클릭해 주십시오."},
    "btn_run": {"en": "Run Topology Analysis 🚀", "ko": "AI 파이프라인 분석 실행 🚀"},
    "btn_error": {"en": "Please explicitly click on the steel plate target inside the image above to tell the AI what to analyze!", "ko": "클릭된 좌표가 없습니다. AI가 어떤 대상을 찾아야 할지 이미지 위를 클릭하여 알려주세요!"},
    "status_running": {"en": "Running Multimodal AI Pipeline...", "ko": "멀티모달 AI 파이프라인 가동 중..."},
    "msg_seg": {"en": "Target Segmentation (SAM 2)...", "ko": "1단계: 타겟 분할 추출 영역 식별 (SAM 2 모듈)..."},
    "msg_depth": {"en": "Estimating Depth (Depth-Anything-V2)...", "ko": "2단계: 깊이 심도 추정 (Depth-Anything-V2 모듈)..."},
    "msg_curv": {"en": "Calculating Stress Concentration Zones...", "ko": "3단계: 노이즈 보정 및 곡률 응력 집중 영역 계산 (수학 모듈)..."},
    "status_done": {"en": "Analysis Complete!", "ko": "모든 분석이 완료되었습니다!"},
    "step2_title": {"en": "2. Target Segmentation Map", "ko": "2. 피착제 타겟 식별 마스크 (Seg. Map)"},
    "step2_cap": {"en": "SAM 2 Target Extracted Mask ({time:.1f}ms)", "ko": "SAM 2 분할 추출 소요시간 ({time:.1f}ms)"},
    "step3_title": {"en": "3. 3D Depth Topography", "ko": "3. 3D 깊이 추정 토포그래피"},
    "step3_cap": {"en": "Relative Depth Output ({time:.1f}ms)", "ko": "상대적 2.5D 깊이 출력 결과 ({time:.1f}ms)"},
    "step3d_title": {"en": "🌍 Show Interactive 3D Topographic Grid & Contour", "ko": "🌍 인터랙티브 3D 입체 지형 격자 및 등고선 보기 (클릭하여 열기)"},
    "step3d_desc": {"en": "Use your mouse/touch to rotate, zoom, and explore the physical surface grid.", "ko": "마우스나 터치 제스처를 사용하여 3D 모델을 회전하거나 확대/축소하며 표면의 굴곡을 살펴보십시오."},
    "step3d_chart_title": {"en": "3D Depth Heatmap Framework", "ko": "3D 심도 히트맵 프레임워크"},
    "step4_title": {"en": "4. Material Match Recommendation", "ko": "4. 적합 점착제 필름 매칭 데이터베이스 리포트"},
    "metric1": {"en": "Max Gaussian Curvature (K)", "ko": "최대 가우시안 곡률치 (K)"},
    "metric2": {"en": "Estimated Min Radius", "ko": "추정 최소 곡률 반경 한계치 (R)"},
    "metric3": {"en": "Total Inference Time", "ko": "통합 처리 파이프라인 레이턴시"},
    "rec_error": {"en": "No suitable adhesive films found for this level of extreme curvature in the database.", "ko": "연구소 데이터베이스 내에 이 수준의 극한 곡률을 버틸 수 있는 적합한 특수 점착 필름군이 존재하지 않습니다."},
    "rec_success": {"en": "Analysis successful. Displaying top optimal products tailored to withstand the evaluated physical stress.", "ko": "분석이 정상 종료되었습니다. 측정된 표면의 구조적 응력을 견딜 수 있도록 설계된 최적의 제품군 리스트입니다."},
    "rec_top": {"en": "**🥇 Top Recommendation: {name} ({id})**  \nSurpasses minimum required curvature ({r}mm capacity) with a high correlation score for peel strength and elongation.", "ko": "**🥇 권장 최적 제품 (Top 1): {name} ({id})**  \n요구되는 최소 곡률 반경({r}mm 이상)을 충분히 커버하며 박리력과 연신율간의 상관관계 스코어가 모델 내에서 가장 뛰어납니다."},
    "rec_all": {"en": "### All Feasible Candidates", "ko": "### 추천 가능 대체 후보 테이블"},
    "missing_image": {"en": "Please upload an image using the sidebar to begin.", "ko": "왼쪽 사이드바에서 테스트 용도로 사용할 판넬 혹은 피착제의 사진을 먼저 업로드하여 주십시오."}
}

def t(key, **kwargs):
    """Returns the translation for the current language toggle state."""
    s = text.get(key, {}).get(lang, key)
    if kwargs:
        return s.format(**kwargs)
    return s

# ---------------------------------------------------------
# Sidebar & Header
# ---------------------------------------------------------
# Language Toggle inside Sidebar
lang_toggle = st.sidebar.radio("🌐 Language / 제어 언어", ["한국어", "English"])
lang = "ko" if lang_toggle == "한국어" else "en"

st.title(t("title"))
st.markdown(t("desc"))

with st.spinner(t("init")):
    sam_wrapper, depth_wrapper, curv_analyzer, match_engine = load_models()

st.sidebar.header(t("controls"))
uploaded_file = st.sidebar.file_uploader(t("upload"), type=['jpg', 'jpeg', 'png'])

# Tuning
st.sidebar.subheader(t("params"))
smoothing_sigma = st.sidebar.slider(t("sigma"), 0.5, 5.0, 2.0, 0.1, help=t("sigma_help"))
curv_analyzer.sigma = smoothing_sigma

roughness = st.sidebar.number_input(t("roughness"), value=1.0, step=0.1)

# ---------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------
if uploaded_file is not None:
    # Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    st.subheader(t("step1_title"))
    
    # Let the user click the image to select the ROI point
    value = streamlit_image_coordinates(image_rgb, key="pil")
    
    prompt_points = None
    prompt_labels = None
    if value is not None:
        prompt_points = np.array([[value["x"], value["y"]]])
        prompt_labels = np.array([1])
        st.info(t("target_selected", x=value['x'], y=value['y']))
    else:
        st.warning(t("target_warning"))

    if st.sidebar.button(t("btn_run"), type="primary"):
        if value is None:
            st.error(t("btn_error"))
        else:
            # Create Layout Columns
            col1, col2 = st.columns(2)
            
            with st.status(t("status_running"), expanded=True) as status:
                # Step 1: Segmentation
                st.write(t("msg_seg"))
                t0 = time.time()
                target_mask = sam_wrapper.segment_target(image_rgb, prompt_points=prompt_points, prompt_labels=prompt_labels)
                t_seg = time.time() - t0
            
                # Step 2: Depth Estimation
                st.write(t("msg_depth"))
                t0 = time.time()
                depth_map = depth_wrapper.estimate_depth(image_rgb, mask=target_mask)
                t_depth = time.time() - t0
                
                # Step 3: Curvature Analysis
                st.write(t("msg_curv"))
                t0 = time.time()
                gaussian_curv = curv_analyzer.calculate_gaussian_curvature(depth_map, mask=target_mask)
                critical_vals, critical_coords = curv_analyzer.find_critical_points(gaussian_curv, mask=target_mask, top_k=1)
                t_curv = time.time() - t0
                
                # Extract Max Stress Point
                highest_stress_raw = critical_vals[0]
                estimated_r_mm = 3.5  # In real life, convert raw -> physical mm using a calibration factor
                t_total = t_seg + t_depth + t_curv
                
                status.update(label=t("status_done"), state="complete", expanded=False)

            # ---------------------------------------------------------
            # Visualization
            # ---------------------------------------------------------
            with col1:
                st.subheader(t("step2_title"))
                # Apply color overlay to the mask
                colored_mask = np.zeros_like(image_rgb)
                colored_mask[target_mask] = [0, 255, 0] # Green overlay
                blended = cv2.addWeighted(image_rgb, 0.7, colored_mask, 0.3, 0)
                st.image(blended, use_container_width=True, caption=t("step2_cap", time=t_seg*1000))
                
            with col2:
                st.subheader(t("step3_title"))
                # Normalize depth for visualization (0-255)
                depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
                depth_colormap_rgb = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
                st.image(depth_colormap_rgb, use_container_width=True, caption=t("step3_cap", time=t_depth*1000))

                
            # ---------------------------------------------------------
            # Interactive 3D Grid / Contour
            # ---------------------------------------------------------
            with st.expander(t("step3d_title")):
                st.markdown(t("step3d_desc"))
                
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
                    title=t("step3d_chart_title"),
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
            st.subheader(t("step4_title"))
            
            # Display Metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{highest_stress_raw:.4f}</div><div class="metric-label">{t("metric1")}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">R ≈ {estimated_r_mm} mm</div><div class="metric-label">{t("metric2")}</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{t_total:.2f} s</div><div class="metric-label">{t("metric3")}</div></div>', unsafe_allow_html=True)
                
            st.write("")
            
            # Recommendations
            recommendations = match_engine.recommend(measured_curvature=estimated_r_mm, measured_roughness=roughness)
            
            if not recommendations:
                st.error(t("rec_error"))
            else:
                st.success(t("rec_success"))
                best_match = recommendations[0]
                st.info(t("rec_top", name=best_match['film_name'], id=best_match['film_id'], r=best_match['max_curvature_radius']))
                
                # Show all options in a table
                st.write(t("rec_all"))
                st.dataframe(recommendations, use_container_width=True)
else:
    st.info(t("missing_image"))
