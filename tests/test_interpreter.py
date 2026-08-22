import os

from render_langgraph import interpreter
from render_langgraph.interpreter import _candidates, resolve_interpreter


def _no_poetry_or_uv(monkeypatch):
    # Isolate from whatever the host machine happens to have installed --
    # poetry/uv presence is environment-dependent and not what these tests
    # are about.
    monkeypatch.setattr(interpreter.shutil, "which", lambda name: None)


def test_candidates_never_include_current_interpreter_first(tmp_path, monkeypatch):
    """Regression: the tool's own interpreter (as it would appear under
    pipx) must not be a candidate at all from _candidates() -- only
    VIRTUAL_ENV/project venvs/poetry/uv are. It's added by
    resolve_interpreter() separately, and only as a last resort."""
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    _no_poetry_or_uv(monkeypatch)
    candidates = _candidates(tmp_path, None)
    sources = [c[0] for c in candidates]
    assert not any("current interpreter" in s for s in sources)
    assert sources == ["./.venv", "./venv", "./env"]


def test_explicit_python_is_the_only_candidate(tmp_path, monkeypatch):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    candidates = _candidates(tmp_path, "C:/somewhere/python.exe")
    assert candidates[0] == ("--python", __import__("pathlib").Path("C:/somewhere/python.exe"))


def test_virtual_env_takes_priority_over_project_venv(tmp_path, monkeypatch):
    fake_venv = tmp_path / "activated"
    scripts_dir = fake_venv / ("Scripts" if os.name == "nt" else "bin")
    scripts_dir.mkdir(parents=True)
    python_name = "python.exe" if os.name == "nt" else "python"
    (scripts_dir / python_name).write_bytes(b"")

    project_venv = tmp_path / ".venv"
    project_scripts = project_venv / ("Scripts" if os.name == "nt" else "bin")
    project_scripts.mkdir(parents=True)
    (project_scripts / python_name).write_bytes(b"")

    monkeypatch.setenv("VIRTUAL_ENV", str(fake_venv))
    candidates = _candidates(tmp_path, None)
    assert candidates[0][0].startswith("$VIRTUAL_ENV")
    assert candidates[0][1] == scripts_dir / python_name


def test_all_candidates_recorded_with_rejection_reasons(tmp_path, monkeypatch):
    """The point of tracking every attempt: a total-failure report can name
    exactly which paths were checked and why each was rejected, not just
    'nothing found'."""
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    _no_poetry_or_uv(monkeypatch)
    result = resolve_interpreter(tmp_path)
    assert result.python is None
    statuses = {a.source: a.status for a in result.attempts}
    assert statuses["./.venv"] == "not_found"
    assert statuses["./venv"] == "not_found"
    assert statuses["./env"] == "not_found"
    # sys.executable (running this test) lacks langgraph -> tried last, rejected
    assert result.attempts[-1].source == "current interpreter (no project venv found)"
    assert result.attempts[-1].status == "import_failed"


def test_sys_executable_fallback_only_when_nothing_found_on_disk(tmp_path, monkeypatch):
    """If a project venv DOES exist on disk but lacks langgraph, resolve_interpreter
    must report that failure -- it must NOT silently fall back to sys.executable,
    which could mask which environment the user actually needs to fix."""
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    _no_poetry_or_uv(monkeypatch)

    venv_dir = tmp_path / ".venv"
    scripts_dir = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    scripts_dir.mkdir(parents=True)
    python_name = "python.exe" if os.name == "nt" else "python"
    fake_python = scripts_dir / python_name
    fake_python.write_bytes(b"not a real interpreter")

    result = resolve_interpreter(tmp_path)
    assert result.python is None
    sources = [a.source for a in result.attempts]
    assert "current interpreter (no project venv found)" not in sources


def test_explicit_python_bypasses_cascade_entirely(target_python, tmp_path, monkeypatch):
    """--python is authoritative: even in a directory with no venv at all,
    passing the real target interpreter explicitly must succeed without
    ever trying project-relative candidates."""
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    _no_poetry_or_uv(monkeypatch)
    result = resolve_interpreter(tmp_path, explicit_python=str(target_python))
    assert result.python == target_python
    assert result.source == "--python"
    assert len(result.attempts) == 1


def test_target_project_venv_is_selected_over_tool_env(target_python, target_project, monkeypatch):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    result = resolve_interpreter(target_project)
    assert result.python is not None
    assert result.python.resolve() == target_python.resolve()
