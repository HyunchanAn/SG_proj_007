import pytest
from fastapi.testclient import TestClient
from api import app, models
import numpy as np


# We can bypass the real models by inserting dummy objects into the `models` dict
class DummyCurvatureAnalyzer:
    def calculate_gaussian_curvature(self, depth, mask):
        return np.array([[1.0, 2.0], [3.0, 4.0]])

    def find_critical_points(self, curv, mask, top_k):
        return [4.0], [(1, 1)]


class DummyKnowledgeEngine:
    def recommend(self, measured_curvature, measured_roughness):
        return [{"film_id": "F-DUMMY", "film_name": "Dummy Film", "match_score": 100}]


class DummyDepthWrapper:
    def estimate_depth(self, img, mask):
        return np.ones((10, 10))


class DummySamWrapper:
    def segment_target(self, img, prompt_points=None, prompt_labels=None):
        return np.ones((10, 10), dtype=bool)


client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_models():
    # Insert dummy models
    models["sam"] = DummySamWrapper()
    models["depth"] = DummyDepthWrapper()
    models["curv"] = DummyCurvatureAnalyzer()
    models["match"] = DummyKnowledgeEngine()
    yield
    # Clean up after tests
    models.clear()


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["models_loaded"] is True


def test_analyze_image_mock():
    # Create a dummy image
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    import cv2

    _, encoded_img = cv2.imencode(".jpg", dummy_img)
    file_bytes = encoded_img.tobytes()

    response = client.post(
        "/api/v1/analyze",
        files={"file": ("dummy.jpg", file_bytes, "image/jpeg")},
        data={"ref_length_mm": 100.0, "roughness": 1.0},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "metrics" in data
    assert "recommendations" in data
    assert len(data["recommendations"]) == 1
    assert data["recommendations"][0]["film_id"] == "F-DUMMY"
