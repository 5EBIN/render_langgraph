from __future__ import annotations

import json
import os
import queue
import socket
import threading
import time
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import as_file, files
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from render_langgraph import discover
from render_langgraph.extract import extract, prepare_target
from render_langgraph.interpreter import resolve_interpreter
from render_langgraph.logging_setup import get_logger
from render_langgraph.spec import split_spec

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:  # pragma: no cover - watchdog is a declared dependency
    FileSystemEventHandler = object  # type: ignore[assignment,misc]
    Observer = None  # type: ignore[assignment]

MAX_PORT_RETRIES = 20

# Reachable by cli.py's signal handlers, which fire asynchronously and have
# no other way to get at whatever serve() is currently running.
_current_handle: "ServerHandle | None" = None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _web_dir() -> Path | None:
    try:
        with as_file(files("render_langgraph") / "web") as p:
            if p.is_dir() and any(p.iterdir()):
                return p
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    return None


class ServerHandle:
    """Owns everything a running `serve()` call needs torn down: the HTTP
    server (and its bound port), the file watcher, and -- indirectly, via
    extract.terminate_all_extractors() -- any extractor subprocess still
    running. cleanup() is idempotent and safe to call from a signal handler,
    atexit, or a normal finally block; whichever gets there first wins."""

    def __init__(self):
        self.httpd: ThreadingHTTPServer | None = None
        self.observer = None
        self.port: int | None = None
        self._lock = threading.Lock()
        self._cleaned_up = False

    def cleanup(self) -> None:
        with self._lock:
            if self._cleaned_up:
                return
            self._cleaned_up = True

        log = get_logger()

        if self.observer is not None:
            try:
                self.observer.stop()
                self.observer.join(timeout=2)
            except Exception:
                log.debug("file watcher didn't stop cleanly")

        from render_langgraph.extract import terminate_all_extractors

        terminate_all_extractors()

        if self.httpd is not None:
            # httpd.shutdown() blocks on an internal Event that only
            # serve_forever()'s own teardown ever sets -- if serve_forever()
            # was never actually entered (an exception during startup, e.g.
            # the file watcher failing to start, before the request loop
            # begins), that Event is never set and shutdown() would hang
            # forever. Run it in a bounded background thread so a stuck
            # coordination wait can never block releasing the port; the
            # actual socket close below is what guarantees the port is free.
            try:
                shutdown_thread = threading.Thread(target=self.httpd.shutdown, daemon=True)
                shutdown_thread.start()
                shutdown_thread.join(timeout=3)
            except Exception:
                log.debug("server shutdown() didn't complete cleanly")
            try:
                self.httpd.server_close()
            except Exception:
                log.debug("server socket didn't close cleanly")

        if self.port is not None:
            log.ok(f"server stopped, port {self.port} released")


@dataclass
class GraphEntry:
    id: str
    label: str
    candidate: discover.Candidate


def _make_entries(candidates: list[discover.Candidate]) -> list[GraphEntry]:
    """One entry per candidate, with a stable, human-readable, and unique
    `id` for API addressing (`/api/graph?graph=<id>`) -- derived from the
    candidate's display name, disambiguated with a numeric suffix in the
    rare case two different files produce the same name."""
    entries = []
    seen: dict[str, int] = {}
    for c in candidates:
        label = discover.display_name(c)
        seen[label] = seen.get(label, 0) + 1
        graph_id = label if seen[label] == 1 else f"{label}-{seen[label]}"
        entries.append(GraphEntry(id=graph_id, label=label, candidate=c))
    return entries


class _State:
    def __init__(self, entries: list[GraphEntry], cwd: Path, explicit_python: str | None = None):
        self.entries = entries
        self.by_id = {e.id: e for e in entries}
        self.cwd = cwd
        self.explicit_python = explicit_python
        self.lock = threading.Lock()
        self.last_hash: dict[str, str | None] = {}
        self.last_good: dict[str, dict] = {}
        self.subscribers: list[queue.Queue] = []

    def graph_list(self) -> list[dict]:
        return [{"id": e.id, "label": e.label} for e in self.entries]

    def default_graph_id(self) -> str | None:
        return self.entries[0].id if self.entries else None

    def run_extract(self, graph_id: str) -> dict:
        entry = self.by_id.get(graph_id)
        if entry is None:
            return {
                "error": f"no such graph '{graph_id}'",
                "kind": "attr_missing",
                "detail": {"available": list(self.by_id)},
            }
        target = prepare_target(entry.candidate.spec, self.cwd)
        interp = resolve_interpreter(target.project_root, explicit_python=self.explicit_python)
        if interp.python is None:
            return {
                "error": "no interpreter with langgraph found",
                "kind": "import_error",
                "detail": {"attempts": [{"source": a.source, "python": str(a.python), "status": a.status} for a in interp.attempts]},
            }
        result = extract(interp.python, entry.candidate.spec, self.cwd)
        return result.data

    def notify_change(self) -> None:
        log = get_logger()
        log.info("change detected, rebuilding graph(s)")
        any_changed = False

        for entry in self.entries:
            data = self.run_extract(entry.id)
            if "error" in data:
                log.warn(f"reload failed for '{entry.id}' ({data.get('kind', 'unknown')}): {data.get('error')} -- keeping last good graph")
                continue  # don't touch last_hash/last_good for this one: the
                # currently-open UI keeps showing the last graph that actually built.

            new_hash = data.get("hash")
            with self.lock:
                changed = new_hash != self.last_hash.get(entry.id)
                self.last_hash[entry.id] = new_hash
                self.last_good[entry.id] = data
            any_changed = any_changed or changed

            n_nodes, n_edges = len(data.get("nodes", [])), len(data.get("edges", []))
            layout_note = "layout recomputed" if changed else "layout preserved"
            log.ok(f"reloaded '{entry.id}': {n_nodes} nodes, {n_edges} edges ({layout_note})")

        payload = json.dumps({"changed": any_changed})
        for q in list(self.subscribers):
            q.put(payload)


class _DebouncedHandler(FileSystemEventHandler):
    def __init__(self, state: _State, debounce: float = 0.3):
        self.state = state
        self.debounce = debounce
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def on_any_event(self, event):
        if event.is_directory or not str(event.src_path).endswith(".py"):
            return
        with self._lock:
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce, self.state.notify_change)
            self._timer.daemon = True
            self._timer.start()


def _make_handler(state: _State, web_dir: Path | None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send_json(self, payload: dict, status: int = 200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/api/graphs":
                self._send_json({"graphs": state.graph_list()}, 200)
                return

            if path == "/api/graph":
                query = parse_qs(parsed.query)
                graph_id = (query.get("graph") or [None])[0] or state.default_graph_id()
                if graph_id is None:
                    self._send_json({"error": "no graphs discovered", "kind": "attr_missing", "detail": {}}, 422)
                    return
                data = state.run_extract(graph_id)
                status = 200 if "error" not in data else 422
                if status == 200:
                    with state.lock:
                        state.last_hash[graph_id] = data.get("hash")
                        state.last_good[graph_id] = data
                self._send_json(data, status)
                return

            if path == "/api/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                q: queue.Queue = queue.Queue()
                state.subscribers.append(q)
                try:
                    while True:
                        try:
                            msg = q.get(timeout=15)
                            self.wfile.write(f"data: {msg}\n\n".encode("utf-8"))
                        except queue.Empty:
                            self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    state.subscribers.remove(q)
                return

            if web_dir is None:
                self._send_json({"error": "frontend not built; use --json to get raw graph data"}, 501)
                return

            rel = path.lstrip("/") or "index.html"
            candidate = (web_dir / rel).resolve()
            if not str(candidate).startswith(str(web_dir.resolve())) or not candidate.is_file():
                candidate = web_dir / "index.html"
            if not candidate.is_file():
                self._send_json({"error": "index.html missing from bundled frontend"}, 404)
                return
            body = candidate.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", _content_type(candidate))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _content_type(path: Path) -> str:
    return {
        ".html": "text/html", ".js": "application/javascript", ".css": "text/css",
        ".json": "application/json", ".svg": "image/svg+xml", ".map": "application/json",
    }.get(path.suffix, "application/octet-stream")


def _bind_server(requested_port: int, handler) -> tuple[ThreadingHTTPServer, int]:
    """If a specific port was requested, try it, then successive ports on
    conflict (logging each move) -- never someone else's process, just the
    next free slot. Falls back to an OS-assigned free port if the requested
    one and its neighbors are all busy."""
    log = get_logger()
    if requested_port:
        port = requested_port
        for _ in range(MAX_PORT_RETRIES):
            try:
                return ThreadingHTTPServer(("127.0.0.1", port), handler), port
            except OSError:
                log.warn(f"port {port} is in use, trying {port + 1}")
                port += 1
        log.warn(f"ports {requested_port}-{port - 1} all in use, letting the OS pick a free one")

    free = _free_port()
    return ThreadingHTTPServer(("127.0.0.1", free), handler), free


def _watch_dir_for(candidate: discover.Candidate, cwd: Path) -> str:
    watch_path, _ = split_spec(candidate.spec)
    if Path(watch_path).is_absolute():
        return str(Path(watch_path).parent)
    return str((cwd / watch_path).resolve().parent)


def serve(
    candidates: discover.Candidate | list[discover.Candidate],
    cwd: Path,
    port: int = 0,
    open_browser: bool = True,
    explicit_python: str | None = None,
) -> int:
    global _current_handle
    log = get_logger()
    log.info("starting server")

    if isinstance(candidates, discover.Candidate):
        candidates = [candidates]
    entries = _make_entries(candidates)
    if len(entries) > 1:
        log.info(f"serving {len(entries)} graphs: {', '.join(e.label for e in entries)}")

    handle = ServerHandle()
    _current_handle = handle

    state = _State(entries, cwd, explicit_python=explicit_python)
    web_dir = _web_dir()
    if web_dir is None:
        log.warn("frontend assets not found in this install; serving API only (/api/graph)")

    http_handler = _make_handler(state, web_dir)
    httpd, bound_port = _bind_server(port, http_handler)
    handle.httpd = httpd
    handle.port = bound_port

    # Everything from here on -- watcher setup, opening a browser,
    # serve_forever() itself -- is covered by the same finally block. The
    # port is bound as of the line above, so any exception past this point
    # must still release it; narrowing this to just wrap serve_forever()
    # would leave the port leaked if, say, the file watcher failed to start.
    try:
        if Observer is not None and entries:
            observer = Observer()
            handler_obj = _DebouncedHandler(state)
            watch_dirs = {_watch_dir_for(e.candidate, cwd) for e in entries}
            for watch_dir in watch_dirs:
                observer.schedule(handler_obj, watch_dir, recursive=True)
            observer.start()
            handle.observer = observer

        url = f"http://127.0.0.1:{bound_port}/"
        log.ok(f"serving on {url} (pid {os.getpid()})")
        if open_browser:
            log.info("opening browser")
            webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    finally:
        handle.cleanup()
        if _current_handle is handle:
            _current_handle = None
    return 0
