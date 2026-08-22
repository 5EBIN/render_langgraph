from __future__ import annotations

import argparse
import atexit
import json
import signal
import sys
import time
from pathlib import Path

from render_langgraph import discover, static_parse
from render_langgraph.extract import extract, prepare_target, terminate_all_extractors
from render_langgraph.interpreter import resolve_interpreter
from render_langgraph.logging_setup import get_logger, setup_logging
from render_langgraph.spec import split_spec

_LAST_ERROR_CACHE = ".render-langgraph/last_error.json"
_LAST_ERROR_TTL_SECONDS = 300


def _format_error(result: dict, verbose: bool) -> None:
    log = get_logger()
    kind = result.get("kind", "unknown")
    message = result.get("error", "unknown error")
    detail = result.get("detail", {})

    if kind == "factory_needs_args":
        callable_name = detail.get("callable", "?")
        log.error(
            f"{message}\n"
            f"  -> retry with the args baked in, or point at a factory with no required args\n"
            f"     e.g. define a zero-arg wrapper and run: render-langgraph path.py:{callable_name}"
        )
    elif kind == "config_error":
        missing = detail.get("missing_env") or []
        need = f"needs: {', '.join(missing)}" if missing else "your app's config failed to load while importing"
        log.error(f"{message}\n  -> {need}\n     set the required environment variables (or add a .env) and retry")
    elif kind == "import_error":
        mod = detail.get("failing_module", "?")
        log.error(
            f"{message}\n"
            f"  -> module '{mod}' isn't importable by name\n"
            f"     is the project installed? try `pip install -e .` or run render-langgraph from the project root"
        )
    elif kind == "attr_missing":
        available = detail.get("available") or []
        hint = f"graph-like objects found instead: {', '.join(available)}" if available else "no graph-like objects were found in the module"
        log.error(f"{message}\n  -> that attribute doesn't exist; {hint}")
    elif kind == "not_a_graph":
        log.error(message)
    elif kind == "timeout":
        culprit = detail.get("likely_culprit")
        culprit_note = f"\n     last thing importing: {culprit}" if culprit else ""
        log.error(
            f"{message}\n"
            f"  -> import is taking too long (likely heavy module-level work, not the target file itself){culprit_note}\n"
            f"     as a next step you can try: render-langgraph --static <target>"
        )
    else:
        log.error(f"{message}\n  -> unexpected failure\n     as a manual fallback you can try: render-langgraph --static <target>")

    traceback_text = result.get("traceback")
    if traceback_text and (verbose or kind == "unknown"):
        log.debug(f"traceback:\n{traceback_text}")


def _save_last_error(project_root: Path, candidate_spec: str, result: dict) -> None:
    cache_dir = project_root / ".render-langgraph"
    try:
        cache_dir.mkdir(exist_ok=True)
        (cache_dir / "last_error.json").write_text(
            json.dumps(
                {
                    "spec": candidate_spec,
                    "kind": result.get("kind"),
                    "error": result.get("error"),
                    "timestamp": time.time(),
                }
            ),
            encoding="utf-8",
        )
    except OSError:
        pass  # best-effort only; never let this block reporting the real error


def _pop_recent_last_error(project_root: Path) -> dict | None:
    path = project_root / _LAST_ERROR_CACHE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        path.unlink(missing_ok=True)
        if time.time() - data.get("timestamp", 0) <= _LAST_ERROR_TTL_SECONDS:
            return data
    except (json.JSONDecodeError, OSError, KeyError):
        pass
    return None


def _prompt_picker(candidates: list[discover.Candidate]) -> discover.Candidate | None:
    log = get_logger()
    log.warn(f"{len(candidates)} candidates found, ambiguous -- pick one:")
    for i, c in enumerate(candidates):
        name = getattr(c, "name", None)
        label = f"{name} -> {c.spec}" if name else c.spec
        print(f"  [{i}] {label} ({c.kind}, source={c.source})", file=sys.stderr)
    try:
        choice = input("> ").strip()
    except EOFError:
        return None
    if not choice.isdigit() or not (0 <= int(choice) < len(candidates)):
        return None
    return candidates[int(choice)]


def _resolve_target(args, project_root: Path) -> discover.Candidate | None:
    log = get_logger()
    if args.target:
        log.debug(f"explicit target given: {args.target}")
    else:
        log.info("searching for a graph (langgraph.json -> AST scan -> cache)")

    result = discover.resolve(explicit=args.target, project_root=project_root, graph_name=args.graph)

    if result.error:
        log.error(f"{result.error}\n  -> pass an explicit target: render-langgraph path/to/graph.py:attr")
        return None

    if result.candidates:
        chosen = _prompt_picker(result.candidates)
        if chosen is None:
            log.error("no selection made")
            return None
        discover.save_cache(project_root, chosen)
        return chosen

    resolved = result.resolved
    if resolved.source != "explicit":
        log.ok(f"found via {resolved.source}: {resolved.spec}")
    if resolved.source in ("ast", "langgraph.json"):
        discover.save_cache(project_root, resolved)
    return resolved


def _resolve_for_server(args, cwd: Path) -> list[discover.Candidate] | None:
    """Server-mode discovery: an explicit target or a single natural
    resolution behaves exactly like _resolve_target (one candidate). A
    genuinely ambiguous bare discovery -- multiple distinct logical graphs,
    once entry-point duplicates (a private fn, its sync helper, its async
    CM wrapper) collapse down to one each -- skips the terminal picker
    entirely and returns all of them: the browser becomes the picker
    instead, via the server's multi-graph mode."""
    log = get_logger()
    if args.target:
        log.debug(f"explicit target given: {args.target}")
        return [discover.Candidate(spec=args.target, source="explicit")]

    log.info("searching for a graph (langgraph.json -> AST scan -> cache)")
    result = discover.resolve(explicit=None, project_root=cwd, graph_name=args.graph)

    if result.error:
        log.error(f"{result.error}\n  -> pass an explicit target: render-langgraph path/to/graph.py:attr")
        return None

    if result.candidates:
        all_graphs = discover.resolve_all_graphs(cwd)
        if len(all_graphs) > 1:
            log.ok(f"found {len(all_graphs)} distinct graphs: {', '.join(discover.display_name(c) for c in all_graphs)}")
            return all_graphs
        # Grouping collapsed this down to a single logical graph after all
        # (the "ambiguity" was just multiple entry points to the same
        # graph) -- resolve normally instead of degrading to a 1-item
        # "multi-graph" server.
        resolved = all_graphs[0] if all_graphs else result.candidates[0]
        discover.save_cache(cwd, resolved)
        return [resolved]

    resolved = result.resolved
    if resolved.source != "explicit":
        log.ok(f"found via {resolved.source}: {resolved.spec}")
    if resolved.source in ("ast", "langgraph.json"):
        discover.save_cache(cwd, resolved)
    return [resolved]


_STATIC_BANNER = """
================ STATIC MODE (--static) ================
This does NOT execute your code. It's a best-effort AST scan of
add_node/add_edge/add_conditional_edges/... calls -- it can't see anything
computed at runtime: dynamic node names, conditional-edge targets without
an explicit path_map, or subgraphs built inside a factory function that
was never called.
If you're here because a normal run failed to import your graph, --static
works around that failure, not through it -- fix the import for the full,
accurate graph.
==========================================================
""".strip("\n")


def _run_static(candidate: discover.Candidate, project_root: Path) -> int:
    log = get_logger()
    log.warn(_STATIC_BANNER)

    prior_failure = _pop_recent_last_error(project_root)
    if prior_failure:
        age = int(time.time() - prior_failure.get("timestamp", time.time()))
        log.warn(
            f"you're here after a failed run {age}s ago: [{prior_failure.get('kind')}] {prior_failure.get('error')}"
        )

    path, _attr = split_spec(candidate.spec)
    result = static_parse.parse_file(path)
    print(json.dumps(result, indent=2))
    if result.get("partial"):
        log.warn(
            "partial result -- some constructs couldn't be resolved statically "
            "(computed edges, conditional edges without path_map, dynamically registered nodes, etc.)"
        )
    return 0


def _format_no_interpreter(attempts, project_root: Path) -> None:
    log = get_logger()
    lines = [f"couldn't find a Python interpreter with `langgraph` installed for {project_root}"]
    for a in attempts:
        if a.status == "not_found":
            lines.append(f"  - {a.source}: not found ({a.python})")
        elif a.status == "import_failed":
            lines.append(f"  - {a.source}: found ({a.python}) but `import langgraph` failed there")
        else:
            lines.append(f"  - {a.source}: {a.python} [{a.status}]")
    if not attempts:
        lines.append("  - no venv candidates found at all (checked $VIRTUAL_ENV, ./.venv, ./venv, ./env, poetry, uv)")
    log.error("\n".join(lines))


def _run_extract(candidate: discover.Candidate, cwd: Path, explicit_python: str | None, verbose: bool) -> tuple[int, dict]:
    target = prepare_target(candidate.spec, cwd)
    interp = resolve_interpreter(target.project_root, explicit_python=explicit_python)
    if interp.python is None:
        _format_no_interpreter(interp.attempts, target.project_root)
        return 1, {}

    result = extract(interp.python, candidate.spec, cwd)
    if not result.ok:
        _format_error(result.data, verbose)
        _save_last_error(target.project_root, candidate.spec, result.data)
        return 1, result.data
    return 0, result.data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="render-langgraph", description="Zero-config LangGraph graph visualizer")
    parser.add_argument("target", nargs="?", default=None, help="path/to/graph.py:attr")
    parser.add_argument("--graph", default=None, help="graph name from langgraph.json when multiple exist")
    parser.add_argument("--python", default=None, help="explicit interpreter to run the target's import in")
    parser.add_argument("--static", action="store_true", help="AST-only mode; never imports the target")
    parser.add_argument("--port", type=int, default=0, help="server port (0 = pick a free port)")
    parser.add_argument("--no-open", action="store_true", help="don't open a browser tab")
    parser.add_argument("--json", action="store_true", help="print graph JSON to stdout and exit, no server")

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="show DEBUG detail: paths, candidates, timings, tracebacks")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="show only warnings and errors")
    return parser


def _make_cleanup():
    """Per-invocation cleanup closure. Deliberately not a module-level
    singleton: main() can run more than once in the same process (embedding,
    tests), and a single shared "already cleaned up" flag would silently
    no-op cleanup on every call after the first."""
    done = {"value": False}

    def cleanup() -> None:
        if done["value"]:
            return
        done["value"] = True
        terminate_all_extractors()
        from render_langgraph import server as server_module

        if server_module._current_handle is not None:
            server_module._current_handle.cleanup()

    return cleanup


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    cleanup = _make_cleanup()
    atexit.register(cleanup)

    def _handle_signal(signum, _frame):
        cleanup()
        sys.exit(128 + signum)

    restore: list[tuple[int, object]] = []
    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            previous = signal.signal(sig, _handle_signal)
            restore.append((sig, previous))
        except (ValueError, OSError):
            pass  # e.g. not the main thread, or unsupported on this platform

    try:
        return _run(args)
    finally:
        cleanup()
        atexit.unregister(cleanup)
        # Restore whatever was installed before us -- main() shouldn't
        # permanently hijack the process's signal disposition beyond its
        # own run (matters for embedding and for repeated in-process calls
        # in tests, which would otherwise silently steal pytest's own
        # SIGINT handling for the rest of the session).
        for sig, previous in restore:
            try:
                signal.signal(sig, previous)
            except (ValueError, OSError):
                pass


def _run(args) -> int:
    cwd = Path.cwd()

    if args.static or args.json:
        candidate = _resolve_target(args, cwd)
        if candidate is None:
            return 1
        if args.static:
            return _run_static(candidate, cwd)
        code, data = _run_extract(candidate, cwd, args.python, args.verbose)
        if code != 0:
            return code
        print(json.dumps(data, indent=2))
        return 0

    # Server mode: bare/ambiguous discovery may resolve to more than one
    # graph (see _resolve_for_server) -- --static/--json above always need
    # exactly one target since neither has a browser to pick in.
    candidates = _resolve_for_server(args, cwd)
    if candidates is None:
        return 1

    if len(candidates) == 1:
        # Single graph: keep the existing fail-fast pre-flight check,
        # reported in the terminal before the server ever starts.
        code, _data = _run_extract(candidates[0], cwd, args.python, args.verbose)
        if code != 0:
            return code
    # Multi-graph: no pre-flight check -- one graph failing to build
    # shouldn't block starting the server for the others; each graph's
    # errors surface individually via /api/graph?graph=<id>.

    from render_langgraph.server import serve

    return serve(candidates, cwd, port=args.port, open_browser=not args.no_open, explicit_python=args.python)


if __name__ == "__main__":
    sys.exit(main())
