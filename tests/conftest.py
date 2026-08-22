from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples" / "simple_agent"
NESTED_PROJECT_DIR = Path(__file__).parent.parent / "examples" / "nested_project"


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / "Scripts" / "python.exe" if sys.platform == "win32" else venv_dir / "bin" / "python"


def _langgraph_importable(python: Path) -> bool:
    if not python.exists():
        return False
    out = subprocess.run([str(python), "-c", "import langgraph"], capture_output=True)
    return out.returncode == 0


@pytest.fixture(scope="session")
def target_project() -> Path:
    return EXAMPLES_DIR


@pytest.fixture(scope="session")
def target_python(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A Python interpreter with `langgraph` installed, pointed at examples/simple_agent.

    Reuses examples/simple_agent/.venv if it's already set up (fast local re-runs);
    otherwise builds a throwaway venv for the test session.
    """
    existing = _venv_python(EXAMPLES_DIR / ".venv")
    if _langgraph_importable(existing):
        return existing

    venv_dir = tmp_path_factory.mktemp("render-langgraph-test-venv")
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    python = _venv_python(venv_dir)
    subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", "langgraph", "langchain-core", "pydantic", "pydantic-settings"],
        check=True,
    )
    return python


@pytest.fixture(scope="session")
def nested_project() -> Path:
    return NESTED_PROJECT_DIR


@pytest.fixture(scope="session")
def nested_project_python(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A Python interpreter with `langgraph` installed, pointed at examples/nested_project."""
    existing = _venv_python(NESTED_PROJECT_DIR / ".venv")
    if _langgraph_importable(existing):
        return existing

    venv_dir = tmp_path_factory.mktemp("render-langgraph-nested-test-venv")
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    python = _venv_python(venv_dir)
    subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", "langgraph", "langchain-core"],
        check=True,
    )
    return python
