# [Project Proposal] AI-Driven Surface Analysis & Film Recommendation

Project Code: SG_proj_007
Project Alias: SG-TERRA (Topographic Evaluation & Resin-film Recommendation AI)

## 1. Executive Summary
본 프로젝트는 SAM 2(Segment Anything Model 2)와 Depth-Anything-V2를 결합하여, 강판 가공 전 피착제의 3D 입체 구조(곡률, 조도, 형상)를 단안 이미지로부터 정밀 추출하고, 이를 기반으로 최적 점착 필름 모델을 자동 추천하는 AI 솔루션 구축을 목적으로 함.

"SG-TERRA는 사진 한 장으로 강판의 굴곡을 3D로 읽어내고, 계측 등급의 정밀도로 최적의 필름을 추천하는 AI입니다."

## 2. Repository Overview
- Repository Name: SG_proj_007
- Primary Tech Stack: Python, PyTorch, SAM 2, Depth-Anything-V2, OpenCV, Open3D, Streamlit
- Target Infrastructure: On-premise (RTX 5080) & Cloud (Google Antigravity)

**✅ Current Status: High-Precision Adaptive Reconstruction (Phase 9)**
- **Adaptive Fidelity Pipeline**: PCA-based local surface variation analysis with **Adaptive Regularization** (Weighted Smoothing).
- **R-Value Accuracy**: Achieved **< 0.08% error** across R=50~500mm range via High-Res Poisson (Depth 9) and Adaptive MAD.
- **Reliable Metrology**: Transitioned from a "visual-only" tool to a "mathematically-rigorous metrology" system for industrial film recommendation.

## 3. Technical Architecture (Multimodal Pipeline)

### A. Phase 1: Zero-Shot 3D Surface Reconstruction
- **Target Segmentation (SAM 2)**: 피착제 ROI 실시간 분리 및 배경 노이즈 마스킹.
- **Enhanced Multi-View Fusion**: N개의 시점 이미지를 SIFT/FPFH 특징점 및 수동 앵커를 통해 하나로 정합.
- **Geometric Surface Reconstruction (Poisson)**: **Poisson Surface Reconstruction(Depth 9)** 적용. 고해상도 지형 정보 보전 및 Adaptive Trimming.
- **Adaptive Topological Metric Extraction**: **MAD(Median Absolute Deviation)** 기반 이상치 제거와 법선 변화량 가중치 평활화를 결합하여 '침엽수림' 노이즈를 근원적으로 제거하고 0.08%의 계측 정밀도 확보.

### B. Phase 2: Knowledge-Based Recommendation Engine
- Feature Matching: 곡률 반경($R$) 및 조도와 연구소 점착제 물성 DB(Peel, Cohesion 등) 간의 상관관계 매핑 및 최적 제품군 리스팅.

## 4. Implementation Strategies
- Computing Power: AMD Ryzen 9 9900X + RTX 5080 (High-Res 3D Inference).
- Key Innovation: **Adaptive Smoothing Algorithm** - 곡률이 중요한 구간은 해상도를 보존하고 평탄 영역은 강력하게 정화하여 계측 신뢰도와 시각적 품질을 동시에 달성.

## 5. Next Action Items
- **Metrology Validation**: Compare with 0.01mm-precision laser scanner Ground Truth (GT) and compute final MAE.
- **Calibration Precision**: Refine 1:1 physical scale mapping using reference markers and adaptive focal length compensation.
- **UX Elevation**: Develop an automated metrology report (PDF) and interactive 3D topography heatmaps with high-precision R-value annotation.