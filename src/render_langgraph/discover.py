"""Target resolution cascade: explicit spec > langgraph.json > AST scan > cache.

First hit wins, no prompt unless genuinely ambiguous.
"""
from __future__ import annotations

import ast
import fnmatch
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from render_langgraph.spec import split_spec

CACHE_DIR = ".render-langgraph"
CACHE_FILE = "config.json"

_ALWAYS_SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv",
    "env", "dist", "build", ".render-langgraph", ".pytest_cache", ".mypy_cache",
    ".tox", "site-packages", "egg-info",
}

_PRIORITY_FILENAMES = {"graph.py": 2, "agent.py": 2, "main.py": 1}


@dataclass
class Candidate:
    spec: str  # "path/to/file.py:attr"
    source: str  # "langgraph.json" | "ast" | "cache" | "explicit"
    kind: str = "compiled"  # "compiled" | "builder" | "factory"
    score: int = 0
    needs_args: bool = False
    # Candidates that are different entry points to the SAME logical graph
    # (a private _compile_x_graph, its sync build_x_graph() test helper, its
    # async x_graph() production entry point) share a group key. Defaults to
    # the candidate's own spec, i.e. "not grouped with anything else" --
    # only scan_ast()'s wrapper detection sets a shared one.
    group: str = ""
    # Display name for a graph-selector UI. Falls back to the spec's attr
    # (see display_name()) when not set -- only langgraph.json entries (a
    # real user-given name) set this today.
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.group:
            self.group = self.spec


@dataclass
class DiscoveryResult:
    resolved: Candidate | None = None
    candidates: list[Candidate] = field(default_factory=list)  # populated when ambiguous
    error: str | None = None


def _cache_path(project_root: Path) -> Path:
    return project_root / CACHE_DIR / CACHE_FILE


def load_cache(project_root: Path) -> Candidate | None:
    path = _cache_path(project_root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Candidate(spec=data["spec"], source="cache", kind=data.get("kind", "compiled"))
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def save_cache(project_root: Path, candidate: Candidate) -> None:
    cache_dir = project_root / CACHE_DIR
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / CACHE_FILE).write_text(
        json.dumps({"spec": candidate.spec, "kind": candidate.kind}, indent=2),
        encoding="utf-8",
    )
    _ensure_gitignored(project_root)


def _ensure_gitignored(project_root: Path) -> None:
    gitignore = project_root / ".gitignore"
    entry = f"{CACHE_DIR}/"
    existing = ""
    if gitignore.is_file():
        existing = gitignore.read_text(encoding="utf-8")
        if entry in existing or CACHE_DIR in existing.splitlines():
            return
    with gitignore.open("a", encoding="utf-8") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        f.write(f"{entry}\n")


def find_langgraph_json(start: Path) -> Path | None:
    current = start.resolve()
    for directory in [current, *current.parents]:
        candidate = directory / "langgraph.json"
        if candidate.is_file():
            return candidate
    return None


def candidates_from_langgraph_json(path: Path) -> list[Candidate]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    graphs = data.get("graphs", {})
    root = path.parent
    out = []
    for name, spec in graphs.items():
        file_part, attr = split_spec(spec)
        abs_spec = f"{(root / file_part).as_posix()}:{attr or 'graph'}"
        out.append(Candidate(spec=abs_spec, source="langgraph.json", kind="compiled", score=10, name=name))
    return out


def display_name(c: Candidate) -> str:
    """Human label for a graph-selector UI: the langgraph.json name if this
    candidate came from one, otherwise the spec's attribute name."""
    if c.name:
        return c.name
    _, attr = split_spec(c.spec)
    return attr or c.spec


def _load_gitignore_patterns(root: Path) -> list[str]:
    gi = root / ".gitignore"
    if not gi.is_file():
        return []
    patterns = []
    for line in gi.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(line.rstrip("/"))
    return patterns


def _is_ignored(rel_path: str, patterns: list[str]) -> bool:
    parts = rel_path.split(os.sep)
    for pattern in patterns:
        if fnmatch.fnmatch(rel_path, pattern) or any(fnmatch.fnmatch(p, pattern) for p in parts):
            return True
    return False


def _enclosing_function(node: ast.AST, tree: ast.Module) -> ast.FunctionDef | None:
    for candidate in ast.walk(tree):
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(candidate):
                if child is node:
                    return candidate
    return None


def _has_required_args(fn: ast.FunctionDef) -> bool:
    args = fn.args
    n_defaults = len(args.defaults)
    n_positional = len(args.args)
    required_positional = n_positional - n_defaults
    required_kwonly = [a for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is None]
    return required_positional > 0 or bool(required_kwonly)


def _is_stategraph_call(value: ast.AST) -> bool:
    return isinstance(value, ast.Call) and (
        (isinstance(value.func, ast.Name) and value.func.id == "StateGraph")
        or (isinstance(value.func, ast.Attribute) and value.func.attr == "StateGraph")
    )


def _stategraph_bound_names(tree: ast.AST) -> set[str]:
    """Names assigned from a `StateGraph(...)` call anywhere in tree (module
    level or inside any function). Used to gate `.compile()` detection --
    see _calls_compile for why that gate matters."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_stategraph_call(node.value):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _calls_compile(node: ast.AST, stategraph_names: set[str]) -> bool:
    """True if a `.compile()` call on a KNOWN StateGraph-bound name appears
    anywhere in node's body -- covers both `x = builder.compile()` and a
    bare `return builder.compile()` (or `yield builder.compile()`), which
    an assignment-only scan misses. This is what lets a private
    `_compile_x_graph(checkpointer)` that never assigns to a local variable
    still be found.

    Restricted to known StateGraph-bound names deliberately: `.compile()`
    alone is far too generic a method name to trust on its own -- it also
    matches `re.compile(pattern)`, a Jinja2 `Environment.compile()`, the
    `compile()` builtin used oddly, etc. Real production code has hit this:
    a module defining `_FENCE_OPEN = re.compile(...)` and an unrelated
    helper function calling `re.compile()` internally both got misdetected
    as compile-producing "graphs" before this gate existed."""
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "compile"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id in stategraph_names
        for n in ast.walk(node)
    )


def _calls_name(node: ast.AST, name: str) -> bool:
    """True if node's body calls a function named `name` anywhere -- used to
    find wrapper functions (a sync `build_x()` test helper, or an async
    `x_graph()` production entry point) around a compile-producing function,
    without caring how deeply nested the call is (inside `async with`, `if`,
    etc.)."""
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name for n in ast.walk(node))


def _score(file_path: Path, root: Path, source: str) -> int:
    score = 0
    if "langgraph" in source:
        score += 3
    score += _PRIORITY_FILENAMES.get(file_path.name, 0)
    try:
        rel = file_path.relative_to(root)
        if rel.parts and rel.parts[0] == "src":
            score += 1
    except ValueError:
        pass
    return score


def scan_ast(root: Path) -> list[Candidate]:
    patterns = _load_gitignore_patterns(root)
    found: list[Candidate] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _ALWAYS_SKIP_DIRS and not d.startswith(".")]
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir != "." and _is_ignored(rel_dir, patterns):
            dirnames[:] = []
            continue

        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            file_path = Path(dirpath) / filename
            rel_file = os.path.relpath(file_path, root)
            if _is_ignored(rel_file, patterns):
                continue

            try:
                source = file_path.read_text(encoding="utf-8")
                tree = ast.parse(source)
            except (SyntaxError, OSError, UnicodeDecodeError):
                continue

            base_score = _score(file_path, root, source)
            stategraph_names = _stategraph_bound_names(tree)

            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                value = node.value
                target_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if not target_names:
                    continue

                is_compile_call = (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "compile"
                    and isinstance(value.func.value, ast.Name)
                    and value.func.value.id in stategraph_names
                )
                is_stategraph_call = _is_stategraph_call(value)
                if not (is_compile_call or is_stategraph_call):
                    continue

                enclosing = _enclosing_function(node, tree)
                for name in target_names:
                    if enclosing is not None:
                        spec = f"{file_path.as_posix()}:{enclosing.name}"
                        found.append(
                            Candidate(
                                spec=spec,
                                source="ast",
                                kind="factory",
                                score=base_score,
                                needs_args=_has_required_args(enclosing),
                            )
                        )
                    else:
                        spec = f"{file_path.as_posix()}:{name}"
                        found.append(
                            Candidate(
                                spec=spec,
                                source="ast",
                                kind="compiled" if is_compile_call else "builder",
                                score=base_score + (1 if is_compile_call else 0),
                            )
                        )

            # Second, broader pass: any function whose body calls .compile()
            # ANYWHERE -- not just as the RHS of an assignment. This is what
            # a private `_compile_x_graph(checkpointer): ... return
            # builder.compile()` needs, since it never assigns to a local
            # variable at all. Deliberately additive rather than a
            # replacement for the loop above: identical specs collapse via
            # the by_spec dedup below, so finding the same function through
            # both passes is harmless.
            #
            # For each one found, also look for OTHER functions in the file
            # that call it by name -- a sync `build_x_graph()` test helper,
            # or an `async def x_graph()` production entry point wrapping it
            # (the common "private compile fn behind an async context
            # manager, with a sync helper for tests" shape). A sync wrapper
            # is scored higher than the raw private function or an async
            # wrapper: it's usually the intended "get a graph with no event
            # loop" entry point, and it's what should win when multiple
            # candidates in this file tie for top score. An async wrapper is
            # still listed (lower score, not hidden) -- an explicit --target
            # at it is still valid; the extractor unwraps it at import time.
            all_funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            for fn in all_funcs:
                if not _calls_compile(fn, stategraph_names):
                    continue
                spec = f"{file_path.as_posix()}:{fn.name}"
                # Shared by this function and all its wrappers below: they're
                # different entry points to the SAME logical graph, and
                # multi-graph mode (see group_by_logical_graph) collapses
                # them back down to one -- the highest-scored -- rather than
                # showing e.g. three tabs for what's really one graph.
                graph_group = f"{file_path.as_posix()}::{fn.name}"
                found.append(
                    Candidate(
                        spec=spec, source="ast", kind="factory", score=base_score,
                        needs_args=_has_required_args(fn), group=graph_group,
                    )
                )
                for wrapper in all_funcs:
                    if wrapper is fn or not _calls_name(wrapper, fn.name):
                        continue
                    wrapper_spec = f"{file_path.as_posix()}:{wrapper.name}"
                    is_sync_wrapper = isinstance(wrapper, ast.FunctionDef)
                    found.append(
                        Candidate(
                            spec=wrapper_spec,
                            source="ast",
                            kind="factory",
                            score=base_score + (5 if is_sync_wrapper else 1),
                            needs_args=_has_required_args(wrapper),
                            group=graph_group,
                        )
                    )

    # de-dup identical specs, keep highest score. On a tie, prefer whichever
    # candidate carries real grouping info: the same function can be found
    # both by the assignment-based scan above (e.g. it also does `builder =
    # StateGraph(...)`, which matches that scan's own unrelated "builder"
    # detection and produces an ungrouped candidate) and by the
    # compile-producer scan below (which sets a shared group with its
    # wrappers). Keeping whichever happened to be inserted first would
    # silently drop the grouping depending on loop order.
    by_spec: dict[str, Candidate] = {}
    for c in found:
        existing = by_spec.get(c.spec)
        if existing is None or c.score > existing.score:
            by_spec[c.spec] = c
        elif c.score == existing.score and existing.group == existing.spec and c.group != c.spec:
            by_spec[c.spec] = c
    return sorted(by_spec.values(), key=lambda c: c.score, reverse=True)


def group_by_logical_graph(candidates: list[Candidate]) -> list[Candidate]:
    """Collapse multiple entry points for the same logical graph (a private
    compile function, its sync test helper, its async CM production entry
    point) down to the single best one -- the sync/async-preferring score
    scan_ast() already assigns picks which. Used for multi-graph mode ("find
    every distinct graph and let the browser pick") so the same underlying
    graph doesn't show up as several redundant entries."""
    best: dict[str, Candidate] = {}
    for c in candidates:
        existing = best.get(c.group)
        if existing is None or c.score > existing.score:
            best[c.group] = c
    return sorted(best.values(), key=lambda c: c.score, reverse=True)


def resolve_all_graphs(project_root: Path) -> list[Candidate]:
    """Every distinct logical graph discoverable in this project, one
    candidate per graph (its best entry point). Used by the server's
    multi-graph mode instead of the single-target `resolve()` cascade when
    discovery would otherwise be ambiguous -- the browser becomes the
    picker instead of a terminal prompt. Does not consult the single-target
    cache; that cache is specifically for "always resolve to this one".
    """
    lg_json = find_langgraph_json(project_root)
    if lg_json:
        candidates = candidates_from_langgraph_json(lg_json)
        if candidates:
            return group_by_logical_graph(candidates)
        return []
    return group_by_logical_graph(scan_ast(project_root))


def resolve(
    explicit: str | None,
    project_root: Path,
    graph_name: str | None = None,
    use_cache: bool = True,
) -> DiscoveryResult:
    if explicit:
        return DiscoveryResult(resolved=Candidate(spec=explicit, source="explicit"))

    if use_cache:
        cached = load_cache(project_root)
        if cached:
            return DiscoveryResult(resolved=cached)

    lg_json = find_langgraph_json(project_root)
    if lg_json:
        candidates = candidates_from_langgraph_json(lg_json)
        if not candidates:
            return DiscoveryResult(error=f"{lg_json} has no usable entries under \"graphs\"")
        if graph_name:
            for c in candidates:
                if getattr(c, "name", None) == graph_name:
                    return DiscoveryResult(resolved=c)
            names = [getattr(c, "name", "?") for c in candidates]
            return DiscoveryResult(error=f"no graph named '{graph_name}' in {lg_json} (have: {', '.join(names)})")
        if len(candidates) == 1:
            return DiscoveryResult(resolved=candidates[0])
        return DiscoveryResult(candidates=candidates)

    ast_candidates = scan_ast(project_root)
    if not ast_candidates:
        return DiscoveryResult(
            error="no langgraph.json and no compiled graph found by scanning *.py files"
        )
    if len(ast_candidates) == 1:
        return DiscoveryResult(resolved=ast_candidates[0])

    top_score = ast_candidates[0].score
    tied = [c for c in ast_candidates if c.score == top_score]
    if len(tied) == 1:
        return DiscoveryResult(resolved=tied[0])
    return DiscoveryResult(candidates=ast_candidates)
