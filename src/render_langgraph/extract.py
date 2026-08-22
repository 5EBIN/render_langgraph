"""Orchestrates running _extractor.py inside the target venv.

This module owns "how do we get from a user-typed spec to something the
extractor subprocess can actually import": splitting path:attr, finding the
project root, and turning a file path into a dotted module name. The
extractor subprocess is handed that already-resolved (project_root,
module_name, attr) triple -- it does not do path math of its own.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from render_langgraph.logging_setup import get_logger
from render_langgraph.spec import split_spec

_EXTRACTOR_PATH = Path(__file__).parent / "_extractor.py"

SOFT_TIMEOUT = 10.0
HARD_TIMEOUT = 60.0

_PROJECT_ROOT_MARKERS = ("pyproject.toml", "setup.py", "setup.cfg")

# Subprocesses currently importing a target module. Tracked at module scope
# (not per-call) so a signal handler or atexit hook anywhere in the process
# can terminate whatever's in flight, regardless of which extract() call
# started it -- server.py's per-request extractions in particular have no
# other natural owner to reach them from during a shutdown.
_active_procs: set[subprocess.Popen] = set()
_active_procs_lock = threading.Lock()


def terminate_all_extractors() -> None:
    """Kill any extractor subprocess still running. Called from cleanup
    paths (signal handlers, atexit) so a shutdown never leaves an orphaned
    import hanging around after render-langgraph itself has exited."""
    with _active_procs_lock:
        procs = list(_active_procs)
    log = get_logger()
    for proc in procs:
        if proc.poll() is None:
            log.debug(f"terminating extractor subprocess pid={proc.pid}")
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass


def find_project_root(start: Path) -> Path:
    """Walk up from `start` (a directory) looking for a project-root anchor
    (pyproject.toml, setup.py, setup.cfg, or .git).

    This is the root that gets put on sys.path and that the target's file
    path is made relative to for the dotted module name -- NOT necessarily
    the directory the user happened to `cd` into. A target file two levels
    deep in a project (e.g. `app/service.py`) still needs its *project's*
    root on sys.path so sibling packages (`app/core`, or a `core` package
    next to `app/`) are importable, not just the directory containing the
    target file itself.

    Falls back to `start` unchanged if no anchor is found (a standalone
    script with no packaging metadata at all) -- in that case the module
    ends up importable as a plain top-level name, which is exactly right
    for a standalone file.
    """
    current = start.resolve()
    for directory in [current, *current.parents]:
        for marker in _PROJECT_ROOT_MARKERS:
            if (directory / marker).is_file():
                return directory
        if (directory / ".git").exists():
            return directory
    return current


def _anchor_name(project_root: Path) -> str:
    """Which marker actually justified this project root, for DEBUG logs."""
    for marker in _PROJECT_ROOT_MARKERS:
        if (project_root / marker).is_file():
            return marker
    if (project_root / ".git").exists():
        return ".git"
    return "no anchor found -- using target file's own directory"


def _dotted_module_name(abs_path: Path, project_root: Path) -> str:
    rel = abs_path.relative_to(project_root) if _is_relative_to(abs_path, project_root) else abs_path
    parts = list(rel.parts)
    if parts and parts[-1].endswith(".py"):
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(p for p in parts if p not in ("", ".", os.sep))


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


@dataclass
class Target:
    project_root: Path
    module_name: str
    attr: str
    file: Path


def prepare_target(spec: str, cwd: Path) -> Target:
    """Resolve a possibly-relative "path:attr" spec into a project root, a
    dotted module name, and an attribute name -- everything the extractor
    subprocess needs, precomputed in the tool's own interpreter."""
    path, attr = split_spec(spec)
    attr = attr or "graph"
    abs_path = (cwd / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    project_root = find_project_root(abs_path.parent)
    module_name = _dotted_module_name(abs_path, project_root)
    return Target(project_root=project_root, module_name=module_name, attr=attr, file=abs_path)


@dataclass
class ExtractResult:
    ok: bool
    data: dict = field(default_factory=dict)


def _relay_subprocess_line(log, line: str) -> None:
    """Raw stdout from the target's own import (plus the extractor's own
    progress prints) is DEBUG-level chatter by default -- it's redundant
    with the INFO line extract() already prints, except for a couple of
    specific events worth surfacing at INFO even when not verbose."""
    if line.startswith("retrying import after loading .env") or line.startswith("factory called:"):
        log.info(line)
    else:
        log.debug(line)


def extract(
    python: Path,
    spec: str,
    cwd: Path,
    xray: bool = True,
    on_progress: Callable[[str], None] | None = None,
    hard_timeout: float = HARD_TIMEOUT,
    soft_timeout: float = SOFT_TIMEOUT,
) -> ExtractResult:
    log = get_logger()
    on_progress = on_progress or (lambda msg: None)
    target = prepare_target(spec, cwd)

    log.info(f"importing {target.module_name} using {python}")
    log.debug(f"project root: {target.project_root} (anchor: {_anchor_name(target.project_root)})")

    with tempfile.TemporaryDirectory() as tmp:
        output_file = os.path.join(tmp, "result.json")
        cmd = [
            str(python),
            "-u",
            str(_EXTRACTOR_PATH),
            str(target.project_root),
            target.module_name,
            target.attr,
            output_file,
            "1" if xray else "0",
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(target.project_root),
        )
        with _active_procs_lock:
            _active_procs.add(proc)

        lines: list[str] = []

        def _pump():
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                lines.append(line)
                _relay_subprocess_line(log, line)
                on_progress(line)

        pump_thread = threading.Thread(target=_pump, daemon=True)
        pump_thread.start()

        try:
            start = time.monotonic()
            warned = False
            timed_out = False
            while True:
                if proc.poll() is not None:
                    break
                elapsed = time.monotonic() - start
                if elapsed > hard_timeout:
                    timed_out = True
                    proc.kill()
                    break
                if elapsed > soft_timeout and not warned:
                    warned = True
                    msg = "still importing (heavy module-level work?)"
                    log.info(msg)
                    on_progress(msg)
                time.sleep(0.1)

            proc.wait()
        finally:
            with _active_procs_lock:
                _active_procs.discard(proc)
        pump_thread.join(timeout=2)

        if timed_out:
            last_target = next(
                (l.split(" ", 1)[1] for l in reversed(lines) if l.startswith("importing ")),
                None,
            )
            return ExtractResult(
                ok=False,
                data={
                    "error": f"import timed out after {hard_timeout:.0f}s",
                    "kind": "timeout",
                    "detail": {"likely_culprit": last_target or target.module_name},
                    "traceback": "\n".join(lines),
                },
            )

        if os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
            with open(output_file, "r", encoding="utf-8") as f:
                result = json.load(f)
            if "error" in result:
                return ExtractResult(ok=False, data=result)
            n_nodes = len(result.get("nodes", []))
            n_edges = len(result.get("edges", []))
            n_subgraphs = len({n["subgraph"] for n in result.get("nodes", []) if n.get("subgraph")})
            log.ok(f"build graph succeeded: {n_nodes} nodes, {n_edges} edges, {n_subgraphs} subgraphs")
            return ExtractResult(ok=True, data=result)

        return ExtractResult(
            ok=False,
            data={
                "error": "extractor exited without producing output",
                "kind": "unknown",
                "detail": {"returncode": proc.returncode},
                "traceback": "\n".join(lines),
            },
        )
