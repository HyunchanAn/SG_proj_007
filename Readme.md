# [Project Proposal] AI-Driven Surface Analysis & Film Recommendation

Project Code: SG_proj_007
Project Alias: SG-TERRA (Topographic Evaluation & Resin-film Recommendation AI)

## 1. Executive Summary
본 프로젝트는 SAM 2(Segment Anything Model 2)와 Depth-Anything-V2를 결합하여, 강판 가공 전 피착제의 3D 입체 구조(곡률, 조도, 형상)를 단안 이미지로부터 정밀 추출하고, 이를 기반으로 프레스 공정 중 들뜸이나 주름 발생을 최소화할 수 있는 최적 점착 필름 모델을 자동 추천하는 AI 솔루션 구축을 목적으로 함.

"SG-TERRA는 사진 한 장으로 강판의 굴곡을 3D로 읽어내고, 우리 회사의 어떤 필름이 가장 완벽하게 버틸 수 있는지 데이터로 답하는 AI입니다."

## 2. Repository Overview
- Repository Name: SG_proj_007
- Primary Tech Stack: Python, PyTorch, SAM 2, Depth-Anything-V2, OpenCV, Streamlit
- Target Infrastructure: On-premise (RTX 5080) & Cloud (Google Antigravity)

**✅ Current Status: MVP Operation Ready**
- End-to-end multimodal pipeline (Segmentation ➡️ Depth ➡️ Curvature ➡️ Knowledge Engine matching) successfully implemented.
- Streamlit interactive UI dashboard (`app.py`) deployed with interactive ROI selection to mitigate background noise.
- redone dashboard interface to support mobile-first single page UI tailored for iOS/iPhone environments, removing tabs and sidebar settings for better accessibility.
- Deployed with hybrid coin auto detection and widget state synchronization to resolve Streamlit cache mismatch bugs.
- **[NEW]** FastAPI 기반의 독립형 REST API 엔드포인트 연동 완료 (Microservice Architecture)

## 3. Installation & Quick Start

### A. Environment Setup
```bash
# Clone the repository
git clone https://github.com/HyunchanAn/SG_proj_007.git
cd SG_proj_007

# Install as a library with development dependencies (pytest, pre-commit, etc.)
pip install -e .[dev]

# Setup pre-commit hooks
pre-commit install
```

### B. Running the Application (Two Modes)
SG-TERRA는 시각적 UI를 위한 **Streamlit 모드**와 타 시스템 연동을 위한 **FastAPI 모드**를 모두 지원합니다.

#### Mode 1: Streamlit Dashboard (UI)
```bash
streamlit run app.py
```
브라우저에서 `http://localhost:8501`에 접속하여 대시보드를 사용할 수 있습니다.

#### Mode 2: FastAPI Headless Server (API)
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
`http://localhost:8000/docs`에 접속하면 Swagger UI를 통해 즉시 이미지를 업로드하고 곡률 분석 및 필름 추천 결과를 JSON으로 테스트할 수 있습니다.

#### API Specification

##### 1. Health Check
* Endpoint: GET /health
* Description: 서버 상태 및 AI 모델 로드 완료 여부를 확인합니다.
* Response:
```json
{
  "status": "ok",
  "models_loaded": true
}
```

##### 2. Analyze Image and Recommend Film
* Endpoint: POST /api/v1/analyze
* Description: 강판 표면 이미지를 분석하여 3D 곡률 반경을 추정하고 최적의 보호 필름을 추천합니다.
* Request Parameters (Multipart Form-Data):
  * file: 분석할 이미지 파일 (필수)
  * ref_length_mm: 물리 크기 변환을 위한 기준 참조 길이 (기본값: 100.0, float)
  * roughness: 강판 표면 조도 Ra (기본값: 1.0, float)
  * click_x: 이미지 내 선택 좌표의 X축 값 (쉼표로 구분된 문자열, 선택사항, 스케일 계산을 위해 두 점 입력 시 쉼표로 구분)
  * click_y: 이미지 내 선택 좌표의 Y축 값 (쉼표로 구분된 문자열, 선택사항)
* Request Example (curl):
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -F "file=@test_image.jpg" \
  -F "ref_length_mm=100.0" \
  -F "roughness=1.0" \
  -F "click_x=120,240" \
  -F "click_y=150,150"
```
* Response:
```json
{
  "status": "success",
  "latency_ms": {
    "segmentation": 120.5,
    "depth": 350.2,
    "curvature": 45.1,
    "matching": 5.4,
    "total": 521.2
  },
  "metrics": {
    "max_gaussian_curvature_raw": 0.0025,
    "estimated_radius_mm": 20.0,
    "critical_point_coords": {
      "x": 180,
      "y": 150
    },
    "pixel_to_mm_scale": 0.8333
  },
  "recommendations": [
    {
      "film_model": "Model-A",
      "suitability": "Highly Recommended",
      "reason": "박리력 및 연신율 조건이 도출된 곡률에 적합함"
    }
  ]
}
```

## 3. Technical Architecture (Multimodal Pipeline)
시스템은 '시각적 형상 파악'과 '물성 매칭'의 두 단계로 구성됨.

### A. Phase 1: Zero-Shot 3D Surface Reconstruction
일반 2D 사진으로부터 고정밀 3D 데이터를 추출하기 위해 최신 Foundation Model을 앙상블함.
- Target Segmentation (SAM 2): 촬영된 이미지 내에서 분석 제외 대상(배경, 노이즈)을 마스킹하고, 순수 강판 표면 영역(ROI)만을 실시간 분리.
- Depth Estimation (Depth-Anything-V2): 단일 시점 이미지에서 픽셀 단위의 상대적 깊이(Relative Depth)를 추정. 보유 중인 RTX 5080(16GB VRAM) 환경을 활용하여 'Large Model' 기반의 고해상도 Depth Map 생성.
- Topological Metric Extraction: 추출된 3D 포인트 클라우드에서 Gaussian Curvature($K$) 및 Surface Area Expansion Ratio를 계산하여 가공 시 응력 집중 구간을 예측.

### B. Phase 2: Knowledge-Based Recommendation Engine
추출된 물리적 지표를 연구소의 점착제 물성 DB와 대조.
- Feature Matching: 곡률 반경($R$) 및 표면 조도($Ra$)와 필름의 박리력(Peel), 유지력(Cohesion), 연신율(Elongation) 간의 상관관계 매핑.
- Optimization: 가공 시 필름이 울지(Wrinkling) 않기 위한 최소 점착력 임계치를 도출하여 최적 제품군 리스팅.

## 4. Core Functional Modules (The 007 Pipeline)
- Module: SEG (Surface Extraction Group): SAM 2 기반으로 피착제 영역 자동 분할. 클램프, 프레스 기계 등 노이즈 마스킹.
- Module: TOPO (Topography Reconstruction): Depth-Anything-V2를 이용한 단안 깊이 추정. 이미지 픽셀 단위의 Z값을 추출하여 피착제의 3D Mesh 데이터 생성.
- Module: CURV (Curvature Analysis): 생성된 3D 지형에서 Gaussian/Mean Curvature 계산. 가공 시 응력 집중(Stress Concentration)이 예상되는 임계 곡률 지점 식별.
- Module: MATCH (Material-Surface Matching): 당사 점착제 물성 DB와 매칭하여, 해당 곡률에서 점착 파괴(Cohesive Failure)가 일어나지 않을 최적의 제품군 추천.

## 5. Implementation Strategies (Hardware & Software)
- Computing Power: 
  - Inference: AMD Ryzen 9 9900X + RTX 5080 (SAM 2 및 Depth 모델 구동용). FP16/BF16 Mixed Precision 추론 모드를 기본 설정으로 채택.
  - Mobility: MacBook Pro M2 자원을 활용한 현장 모니터링 및 클라이언트 인터페이스 구축.
- Architecture Consideration: 로컬 Inference 우선, 대규모 연산 시에만 Antigravity 인스턴스를 활용하는 하이브리드 아키텍처.
- Key Challenges & Mitigation:
  - Specular Reflection: 금속 표면 반사광으로 인한 오차는 Normal Map 정규화 알고리즘을 통해 보정.
  - Scale Calibration: 정확한 곡률 수치 도출을 위해 이미지 내 참조 마커(Reference Marker)를 활용한 단위 환산 적용.

## 6. Expected Benefits (ROI)
- 연구 효율 증대: 수동으로 진행하던 피착제 적합성 테스트를 AI 기반 시뮬레이션으로 대체하여 R&D 리드타임 40% 단축.
- 클레임 방지: 고객사 공정 조건(곡률, 압력)에 최적화된 필름을 데이터 기반으로 제안함으로써 제품 불량률(들뜸, 주름) 최소화.
- 데이터 자산화: 제품 추천 이력을 DB화하여 차세대 점착제 조성 설계(Formulation)의 기초 자료로 활용.

## 7. Next Action Items
- Dataset Setup: 연구소 보유 필름 물성 데이터 시트(CSV) 연동 API 개발.
- Model Fine-tuning: 금속 표면 특화 뎁스 추정을 위한 합성 데이터(Synthetic Data) 생성 및 학습 파이프라인 구축.
- Latency Optimization: SAM 2 및 Depth-Anything-V2 파이프라인의 Inference Latency 최소화.