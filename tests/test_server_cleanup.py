"""Hard requirement: no exit path may leak the server's port. Covers the
three ways a process can end: normal completion, SIGINT, and an unhandled
exception -- each must release the port."""
from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest

from render_langgraph import cli
from render_langgraph import discover
from render_langgraph import server as server_module


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def test_normal_exit_releases_port(tmp_path):
    (tmp_path / "graph.py").write_text("graph = None\n", encoding="utf-8")
    candidate = discover.Candidate(spec=f"{(tmp_path / 'graph.py').as_posix()}:graph", source="explicit")

    outcome = {}

    def run():
        outcome["code"] = server_module.serve(candidate, tmp_path, port=0, open_browser=False)

    t = threading.Thread(target=run, daemon=True)
    t.start()

    handle = None
    for _ in range(50):
        handle = server_module._current_handle
        if handle is not None and handle.port is not None:
            break
        time.sleep(0.1)
    assert handle is not None and handle.port is not None, "server never finished starting"

    port = handle.port
    assert not _port_is_free(port)  # actively bound while serving

    handle.httpd.shutdown()  # simulate normal completion: serve_forever() returns
    t.join(timeout=5)
    assert not t.is_alive()
    assert outcome.get("code") == 0

    assert _port_is_free(port), "port was not released after normal exit"


def test_unhandled_exception_still_releases_port(tmp_path, monkeypatch):
    (tmp_path / "graph.py").write_text("graph = None\n", encoding="utf-8")
    candidate = discover.Candidate(spec=f"{(tmp_path / 'graph.py').as_posix()}:graph", source="explicit")

    class Boom(Exception):
        pass

    def boom_serve_forever(self):
        raise Boom("simulated crash")

    monkeypatch.setattr(server_module.ThreadingHTTPServer, "serve_forever", boom_serve_forever)

    captured: dict = {}
    original_bind = server_module._bind_server

    def spy_bind(requested_port, handler):
        httpd, port = original_bind(requested_port, handler)
        captured["port"] = port
        return httpd, port

    monkeypatch.setattr(server_module, "_bind_server", spy_bind)

    with pytest.raises(Boom):
        server_module.serve(candidate, tmp_path, port=0, open_browser=False)

    assert "port" in captured
    assert _port_is_free(captured["port"]), "port was not released after an unhandled exception"


def test_sigint_releases_port_in_process(target_python, target_project):
    """Exercises the real signal.signal(SIGINT, ...) handler installed by
    cli.main() -- no subprocess/console needed, since Python always runs a
    registered signal handler on the main thread regardless of which
    thread actually called raise_signal(). cli.main() must therefore run
    on pytest's own main thread here; a second thread waits for the server
    to come up and then raises the signal at it."""
    captured: dict = {}

    def trigger_once_serving():
        handle = None
        for _ in range(100):
            handle = server_module._current_handle
            if handle is not None and handle.port is not None:
                break
            time.sleep(0.1)
        assert handle is not None, "server never started"
        captured["port"] = handle.port
        assert not _port_is_free(handle.port)
        signal.raise_signal(signal.SIGINT)

    trigger_thread = threading.Thread(target=trigger_once_serving, daemon=True)
    trigger_thread.start()

    old_cwd = os.getcwd()
    os.chdir(target_project)
    try:
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["linear_graph.py:graph", "--no-open", "--python", str(target_python)])
    finally:
        os.chdir(old_cwd)

    trigger_thread.join(timeout=5)
    assert not trigger_thread.is_alive()
    assert exc_info.value.code == 128 + signal.SIGINT
    assert "port" in captured, "trigger thread never observed the server starting"
    assert _port_is_free(captured["port"]), "port was not released after SIGINT"


@pytest.mark.skipif(
    os.name == "nt" and not sys.stdout.isatty(),
    reason="CTRL_C_EVENT delivery on Windows needs a real console attached to this process",
)
def test_sigint_releases_port_subprocess(target_python, target_project):
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen(
        [sys.executable, "-m", "render_langgraph", "linear_graph.py:graph", "--no-open", "--python", str(target_python)],
        cwd=str(target_project),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=creationflags,
    )
    try:
        port = None
        buffer_lines: list[str] = []
        deadline = time.time() + 30
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            buffer_lines.append(line)
            m = re.search(r"serving on http://127\.0\.0\.1:(\d+)/", line)
            if m:
                port = int(m.group(1))
                break

        assert port is not None, f"server never reported it started; output so far:\n{''.join(buffer_lines)}"
        assert not _port_is_free(port)

        if os.name == "nt":
            proc.send_signal(signal.CTRL_C_EVENT)
        else:
            proc.send_signal(signal.SIGINT)

        proc.wait(timeout=20)

        released = False
        for _ in range(50):
            if _port_is_free(port):
                released = True
                break
            time.sleep(0.1)
        assert released, "port was not released after SIGINT"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
