import pandas as pd
from typing import Dict, List, Optional

class KnowledgeEngine:
    def __init__(self, db_path: str):
        """
        초기화 및 물성 데이터 로드.
        :param db_path: 점착제 물성 CSV 파일 경로
        """
        self.db_path = db_path
        self.db = pd.DataFrame()
        self._load_database()

    def _load_database(self):
        """내장된 CSV 파일로부터 물성 데이터를 로드합니다."""
        try:
            self.db = pd.read_csv(self.db_path)
            print(f"DB Loaded: {len(self.db)} film records from {self.db_path}")
        except FileNotFoundError:
            print(f"Warning: DB file not found at {self.db_path}")
            print("초기 개발 테스트를 위해 Mock 데이터를 생성합니다.")
            self.db = pd.DataFrame({
                'film_id': ['F-001', 'F-002', 'F-003'],
                'film_name': ['Standard Acrylic', 'High-Adhesion Silicone', 'Ultra-Flexible TPU'],
                'peel_strength': [15.0, 25.0, 18.0],     # N/25mm
                'cohesion': [2000, 3500, 2500],          # minutes
                'elongation': [150, 80, 400],            # %
                'max_curvature_radius': [10.0, 5.0, 2.0] # 허용 가능한 최소 곡률 반경 (mm) - 숫자가 작을수록 급격한 곡면 커버 가능
            })

    def recommend(self, measured_curvature: float, measured_roughness: float) -> List[Dict]:
        """
        추출된 강판의 곡률 반경(R)과 조도(Ra)를 바탕으로 최적의 필름을 추천.
        :param measured_curvature: 측정된 곡률 반경 (mm) - 가장 급격히 꺾인 R값
        :param measured_roughness: 측정된 표면 조도 (Ra)
        :return: 추천된 필름 목록 (딕셔너리 리스트)
        """
        if self.db.empty:
            return []

        # 1. 형태 적합성 (Formability): 필름이 커버할 수 있는 곡률 반경(R)보다 측정된 반경이 크거나 같아야 들뜨지 않음.
        # 즉, 필름의 허용 곡률 반경 <= 강판의 곡률 반경
        candidate_films = self.db[self.db['max_curvature_radius'] <= measured_curvature].copy()

        # 2. 우선순위 정렬: 박리력(Peel Strength)과 연신율(Elongation)의 조합 점수를 부여하여 Sorting
        # (예시 로직: 높은 점착력과 충분한 연신율을 가질수록 가공 중 주름/들뜸 방지에 유리)
        candidate_films['match_score'] = candidate_films['peel_strength'] * 0.6 + candidate_films['elongation'] * 0.4
        
        # 점수 기준 내림차순 정렬
        sorted_candidates = candidate_films.sort_values(by='match_score', ascending=False)
        
        results = sorted_candidates.to_dict(orient='records')
        return results

# 테스트 블럭 (직접 실행 시)
if __name__ == "__main__":
    engine = KnowledgeEngine("../../data/database/film_properties.csv")
    # 예시: 가장 급격한 곡률 반경이 3.5mm로 측정된 강판
    recommendations = engine.recommend(measured_curvature=3.5, measured_roughness=1.2)
    print("\n[Recommendation Results for R=3.5mm, Ra=1.2]")
    for idx, film in enumerate(recommendations, 1):
        print(f"{idx}. {film['film_name']} (Score: {film['match_score']:.1f})")
