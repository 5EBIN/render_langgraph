"""Find the Python interpreter that owns the target project's venv.

We never import the user's graph in *our* interpreter. We shell out to
whichever interpreter actually has their dependencies installed -- doing
otherwise (e.g. site.addsitedir() into the tool's own process) risks a
straight-up segfault: compiled extensions like pydantic-core or numpy are
built for a specific Python build, and loading the project's .so/.pyd into
render-langgraph's differently-versioned interpreter is undefined behavior,
not just an import error.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from render_langgraph.logging_setup import get_logger


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


@dataclass
class Attempt:
    source: str
    python: Path
    status: str  # "ok" | "not_found" | "import_failed"
    detail: str = ""


def _candidates(project_root: Path, explicit_python: str | None) -> list[tuple[str, Path]]:
    """Yield (source_label, python_path) candidates in priority order.

    Deliberately does NOT put "the interpreter running render-langgraph itself"
    first. When render-langgraph is pipx-installed, sys.executable/sys.prefix is
    the *tool's own* isolated env, not the target project's -- treating it
    as a first-class candidate risks silently running against the wrong
    (or an empty) environment. It's added by resolve_interpreter() only as
    a last resort, and only if the project has no venv at all.
    """
    found: list[tuple[str, Path]] = []

    # 0. explicit --python override
    if explicit_python:
        found.append(("--python", Path(explicit_python)))

    # 1. $VIRTUAL_ENV
    venv_env = os.environ.get("VIRTUAL_ENV")
    if venv_env:
        found.append((f"$VIRTUAL_ENV ({venv_env})", _venv_python(Path(venv_env))))

    # 2. <project>/.venv, <project>/venv, <project>/env
    for name in (".venv", "venv", "env"):
        found.append((f"./{name}", _venv_python(project_root / name)))

    # 3. poetry
    poetry = shutil.which("poetry")
    if poetry:
        try:
            out = subprocess.run(
                [poetry, "env", "info", "-p"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if out.returncode == 0 and out.stdout.strip():
                found.append(("poetry env", _venv_python(Path(out.stdout.strip()))))
        except (subprocess.TimeoutExpired, OSError):
            pass

    # 4. uv
    uv = shutil.which("uv")
    if uv:
        try:
            out = subprocess.run(
                [uv, "python", "find"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if out.returncode == 0 and out.stdout.strip():
                found.append(("uv python find", Path(out.stdout.strip())))
        except (subprocess.TimeoutExpired, OSError):
            pass

    return found


@dataclass
class InterpreterResult:
    python: Path | None
    source: str | None
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def tried(self) -> list[tuple[str, Path]]:
        """Back-compat view: (source, path) for every candidate attempted."""
        return [(a.source, a.python) for a in self.attempts]

    @property
    def langgraph_missing_from(self) -> list[str]:
        return [f"{a.source} ({a.python})" for a in self.attempts if a.status == "import_failed"]


def resolve_interpreter(project_root: Path, explicit_python: str | None = None) -> InterpreterResult:
    """Find a Python interpreter with `langgraph` importable.

    Tries candidates in priority order, verifying each by actually running
    `import langgraph` in it with cwd=project_root. Records every candidate
    attempted -- including ones that don't exist on disk -- with a reason,
    so failure messages can be precise instead of "couldn't find anything".

    If --python is given explicitly, only that interpreter is tried: an
    explicit override that turns out to lack langgraph should be reported
    plainly, not silently skipped in favor of an auto-discovered one the
    user didn't ask for.

    Falls back to the interpreter running render-langgraph itself (sys.executable)
    only if the project has no venv/poetry/uv candidate found on disk at all.
    """
    log = get_logger()
    log.info(f"resolving interpreter for {project_root}")

    if explicit_python:
        candidates = [("--python", Path(explicit_python))]
    else:
        candidates = _candidates(project_root, None)

    attempts: list[Attempt] = []
    any_found_on_disk = False

    for source, python in candidates:
        if not python.exists():
            log.debug(f"candidate {source} -> {python}: not found")
            attempts.append(Attempt(source, python, "not_found"))
            continue
        any_found_on_disk = True
        log.debug(f"candidate {source} -> {python}: checking `import langgraph`")
        ok, err, version = _verify_langgraph(python, project_root)
        if ok:
            attempts.append(Attempt(source, python, "ok"))
            version_note = f" (python {version})" if version else ""
            log.ok(f"using {source}: {python}{version_note}")
            return InterpreterResult(python=python, source=source, attempts=attempts)
        log.debug(f"candidate {source} -> {python}: import langgraph failed: {err.strip()}")
        attempts.append(Attempt(source, python, "import_failed", err))

    if explicit_python:
        log.error(f"--python {explicit_python} does not have `langgraph` installed")
        return InterpreterResult(python=None, source=None, attempts=attempts)

    if not any_found_on_disk:
        fallback = Path(sys.executable)
        log.warn(
            f"no project venv found under {project_root}; "
            f"falling back to render-langgraph's own interpreter ({fallback}), which likely lacks your project's dependencies"
        )
        ok, err, version = _verify_langgraph(fallback, project_root)
        status = "ok" if ok else "import_failed"
        attempts.append(Attempt("current interpreter (no project venv found)", fallback, status, err))
        if ok:
            version_note = f" (python {version})" if version else ""
            log.ok(f"using current interpreter: {fallback}{version_note}")
            return InterpreterResult(
                python=fallback, source="current interpreter (no project venv found)", attempts=attempts
            )

    return InterpreterResult(python=None, source=None, attempts=attempts)


def _verify_langgraph(python: Path, project_root: Path) -> tuple[bool, str, str | None]:
    try:
        out = subprocess.run(
            [str(python), "-c", "import sys, langgraph; print(sys.version.split()[0])"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=15,
        )
        version = out.stdout.strip() or None
        return out.returncode == 0, out.stderr, version if out.returncode == 0 else None
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, str(exc), None
