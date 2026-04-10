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

**✅ Current Status: Advanced 3D Reconstruction Integration (Phase 7)**
- **High-Fidelity Pipeline**: Unified monocular depth estimation with **32-bit Dynamic Bilateral Filtering** and **Poisson Surface Reconstruction**.
- **Multi-View Fusion**: Adaptive FPFH + RANSAC registration with intelligent reference selection (Sharpness/Entropy based).
- **Interactive UI**: Streamlit dashboard updated with registration fitness guards and optimized 3D topographic grid visualization.
- **Ongoing Challenge**: Tackling high-amplitude spike noise on low-texture surfaces (e.g., white plastic) via Phase 8 radical smoothing.

## 3. Technical Architecture (Multimodal Pipeline)
시스템은 '시각적 형상 파악'과 '물성 매칭'의 두 단계로 구성됨.

### A. Phase 1: Zero-Shot 3D Surface Reconstruction
일반 2D 사진으로부터 고정밀 3D 데이터를 추출하기 위해 최신 Foundation Model을 앙상블함.
- Target Segmentation (SAM 2): 촬영된 이미지 내에서 분석 제외 대상(배경, 노이즈)을 마스킹하고, 순수 강판 표면 영역(ROI)만을 실시간 분리.
- Depth Estimation (Depth-Anything-V2): 단일 시점 이미지에서 픽셀 단위의 상대적 깊이(Relative Depth)를 추정. 보유 중인 RTX 5080(16GB VRAM) 환경을 활용하여 'Large Model' 기반의 고해상도 Depth Map 생성.
- **Enhanced Multi-View Fusion**: N개의 시점 이미지를 특징점(SIFT/FPFH) 및 수동 앵커를 통해 하나로 정합. **Intelligent Reference Selection**을 통해 가장 선명한 이미지를 기준으로 스케일을 정렬(Histogram Matching).
- **Geometric Surface Reconstruction (Poisson)**: 단순 점군을 넘어 포인트 클라우드로부터 연속적 표면 함수를 근사하는 **Poisson Surface Reconstruction(Depth 9)** 적용. 고해상도 옥트리를 통해 미세 곡률 정보를 보전하면서도 **Adaptive Trimming**으로 외곽 노이즈 차단. 
- **Topological Metric Extraction**: **MAD(Median Absolute Deviation)** 기반의 강건한 이상치 제거와 **5x5 Median Blur** 전처리를 결합하여 '침엽수림' 스파이크 노이즈를 근본적으로 제거. 3.3% 이내의 낮은 왜곡률로 고정밀 Gaussian Curvature($K$) 및 곡률 반경(R) 산출.

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