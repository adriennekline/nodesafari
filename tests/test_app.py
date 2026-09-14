from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_getting_started_and_application_navigation_render():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=120)
    assert not app.exception

    labels = [tab.label for tab in app.tabs]
    applications = ["Getting Started", "Data & QC", "Explore", "Compare", "Perturb", "Predict"]
    positions = [labels.index(application) for application in applications]
    assert positions == sorted(positions)
    assert not any("Analysis Navigator" in expander.label for expander in app.expander)
    assert any(selectbox.label == "What do you want to learn?" for selectbox in app.selectbox)
