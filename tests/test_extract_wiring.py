"""Structural checks that don't need a real venv: the extractor subprocess
must be spawned under the interpreter it was explicitly given, never
sys.executable, and a failed interpreter resolution must never trigger an
AST/--static fallback anywhere in the call chain."""
import sys
from pathlib import Path

import render_langgraph.extract as extract_module
from render_langgraph import interpreter as interpreter_module
from render_langgraph import static_parse
from render_langgraph.cli import main as cli_main
from render_langgraph.extract import extract


class _FakeProc:
    captured_cmd = None
    captured_cwd = None

    def __init__(self, cmd, **kwargs):
        _FakeProc.captured_cmd = cmd
        _FakeProc.captured_cwd = kwargs.get("cwd")
        self.stdout = iter(())
        self.returncode = 0

    def poll(self):
        return 0

    def wait(self):
        return 0

    def kill(self):
        pass


def test_subprocess_argv0_is_resolved_interpreter_not_sys_executable(tmp_path, monkeypatch):
    (tmp_path / "graph.py").write_text("graph = None\n", encoding="utf-8")
    monkeypatch.setattr(extract_module.subprocess, "Popen", _FakeProc)

    fake_python = Path("Z:/not-sys-executable/python.exe")
    extract(fake_python, "graph.py:graph", tmp_path)

    assert _FakeProc.captured_cmd[0] == str(fake_python)
    assert _FakeProc.captured_cmd[0] != sys.executable
    assert _FakeProc.captured_cwd == str(tmp_path.resolve())


def test_no_interpreter_never_triggers_ast_fallback(tmp_path, monkeypatch, capsys):
    """cli.main() with a project that has no discoverable venv must fail
    cleanly (interpreter resolution fails) and must NEVER call
    static_parse.parse_file as an automatic substitute -- --static is
    opt-in only."""
    (tmp_path / "graph.py").write_text("graph = None\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(interpreter_module.shutil, "which", lambda name: None)

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("static_parse.parse_file must not run as an automatic fallback")

    monkeypatch.setattr(static_parse, "parse_file", _must_not_be_called)

    code = cli_main(["graph.py:graph", "--json"])
    assert code != 0
