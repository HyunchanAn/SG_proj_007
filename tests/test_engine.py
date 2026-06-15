import pytest

from sg_terra.match.engine import KnowledgeEngine


@pytest.fixture
def mock_engine():
    # Provide a non-existent path to trigger the fallback mock DB logic
    engine = KnowledgeEngine("data/database/non_existent.csv")
    return engine


def test_engine_initialization(mock_engine):
    """Test if the mock engine initialized properly."""
    assert not mock_engine.db.empty
    assert len(mock_engine.db) == 3
    assert "film_id" in mock_engine.db.columns


def test_engine_recommendation_valid(mock_engine):
    """Test recommendation with valid curvature that fits all mock films."""
    # mock films max_curvature_radius are: 10.0, 5.0, 2.0
    # A measured curvature of 15.0 mm means it's a very flat curve.
    # All films (with max_curvature_radius <= 15.0) should be recommended.
    recommendations = mock_engine.recommend(
        measured_curvature=15.0, measured_roughness=1.0
    )
    assert len(recommendations) == 3
    # Check if they are sorted by match_score descending
    assert recommendations[0]["match_score"] >= recommendations[1]["match_score"]


def test_engine_recommendation_extreme(mock_engine):
    """Test recommendation with extreme curvature (very small radius)."""
    # A measured curvature of 1.0 mm (very sharp corner).
    # No mock films have max_curvature_radius <= 1.0 (min is 2.0).
    recommendations = mock_engine.recommend(
        measured_curvature=1.0, measured_roughness=1.0
    )
    assert len(recommendations) == 0


def test_engine_recommendation_partial(mock_engine):
    """Test recommendation where only some films match."""
    # Measured curvature of 3.5 mm.
    # Only films with max_curvature_radius <= 3.5 will match.
    # Mock DB: TPU is 2.0 (matches), Silicone is 5.0 (does not match), Acrylic is 10.0 (does not match).
    recommendations = mock_engine.recommend(
        measured_curvature=3.5, measured_roughness=1.0
    )
    assert len(recommendations) == 1
    assert recommendations[0]["film_id"] == "F-003"
