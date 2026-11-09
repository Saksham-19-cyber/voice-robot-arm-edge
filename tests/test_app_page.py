"""Module."""
import os
import sys

if sys.platform == "win32":
    os.environ.setdefault("MUJOCO_GL", "glfw")
else:
    os.environ.setdefault("MUJOCO_GL", "osmesa")

from streamlit.testing.v1 import AppTest


def test_app_page_initial_render():
    at = AppTest.from_file("app/app.py", default_timeout=30).run()
    assert not at.exception, f"AppTest raised unexpected exception: {at.exception}"
    assert len(at.title) > 0 or len(at.header) > 0
    assert at.sidebar.radio[0].value == "🎮 Interactive Demo"


def test_app_results_and_architecture_pages():
    at = AppTest.from_file("app/app.py", default_timeout=30).run()
    assert not at.exception

    at.sidebar.radio[0].set_value("📊 Benchmarks & Results").run()
    assert not at.exception

    at.sidebar.radio[0].set_value("📐 Architecture & Notes").run()
    assert not at.exception
