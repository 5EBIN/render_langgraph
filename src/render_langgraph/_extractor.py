"""Runs INSIDE the target project's venv. Never imported by render-langgraph itself.

Invoked as:
    {python} _extractor.py <project_root> <module_name> <attr> <output_file> <xray:0|1>

`project_root` and `module_name` are already resolved by extract.py (in the
tool's own interpreter) -- this script does no path math of its own. It
inserts project_root onto sys.path and imports `module_name` BY DOTTED NAME
(importlib.import_module), never by file path. Importing by file path
(spec_from_file_location) loads the file as a parentless top-level module,
which breaks both relative imports and absolute intra-project imports like
`from app.core import x` -- exactly the bug this script exists to avoid.
spec_from_file_location is kept only as a narrow last resort (see
_import_target) for the case where the dotted name itself can't be found.

Writes exactly one JSON document to <output_file> (success schema or the
error contract) and exits 0 regardless of outcome -- the *file* carries the
result, not the process exit code or stdout, because the target module may
print arbitrary things at import time.

Progress lines ("importing <target>") are written to real stdout so the
parent process can relay them live; they are never parsed as data.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import re
import sys
import textwrap
import traceback
import typing


class ExtractError(Exception):
    def __init__(self, kind: str, message: str, detail: dict | None = None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.detail = detail or {}


_MISSING = object()
_ENV_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
_PYDANTIC_MISSING_FIELD_RE = re.compile(r"^(\w+)\n\s*Field required", re.MULTILINE)
_VIRTUAL_NODES = {"__start__": "start", "__end__": "end"}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_dotenv(project_root: str) -> bool:
    path = os.path.join(project_root, ".env")
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        return True
    except OSError:
        return False


def _import_by_file_location(path: str, mod_name: str):
    """Brittle last resort: load a file directly, bypassing dotted import.

    Used only when the dotted module name itself can't be found (not when
    some transitively-imported *other* package is missing) -- see the
    ModuleNotFoundError.name check in _import_target. A module loaded this
    way has no parent package, so its own relative imports would still
    break; this exists purely so a genuinely standalone file with an
    unusual name still has a chance to load.
    """
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _looks_config_related(exc: BaseException) -> bool:
    type_name = type(exc).__name__
    msg = str(exc)
    if "ValidationError" in type_name:
        return True
    if isinstance(exc, (KeyError, EnvironmentError)) and _ENV_TOKEN_RE.search(msg):
        return True
    if "field required" in msg.lower() or "environment variable" in msg.lower():
        return True
    return False


def _extract_missing_env(exc: BaseException) -> list[str]:
    msg = str(exc)
    fields = _PYDANTIC_MISSING_FIELD_RE.findall(msg)
    if fields:
        return [f.upper() for f in dict.fromkeys(fields)][:10]
    tokens = _ENV_TOKEN_RE.findall(msg)
    seen: list[str] = []
    for t in tokens:
        if t not in seen and t not in ("URL", "HTTP", "HTTPS", "JSON", "API"):
            seen.append(t)
    return seen[:10]


def _import_target(project_root: str, mod_name: str, dotenv_loaded: list[bool]):
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    _log(f"importing {mod_name} (root={project_root})")

    caught: BaseException | None = None
    try:
        return importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        # Only fall back to file-path loading when the dotted name ITSELF
        # couldn't be found -- not when some transitively-imported *other*
        # package (fastapi, a project dependency, etc.) is what's missing.
        # Conflating the two used to make a missing dependency look like a
        # sys.path problem.
        missing_top = (e.name or "").split(".")[0]
        target_top = mod_name.split(".")[0]
        if missing_top == target_top:
            file_path = os.path.join(project_root, *mod_name.split(".")) + ".py"
            try:
                mod = _import_by_file_location(file_path, mod_name)
                if mod is not None:
                    return mod
                caught = e
            except Exception as e2:
                caught = e2
        else:
            caught = e
    except Exception as e:
        caught = e

    exc = caught

    if not dotenv_loaded[0] and _looks_config_related(exc):
        dotenv_loaded[0] = True
        if _load_dotenv(project_root):
            _log("retrying import after loading .env")
            try:
                return importlib.import_module(mod_name)
            except Exception as retry_exc:
                exc = retry_exc

    if _looks_config_related(exc):
        raise ExtractError(
            "config_error",
            f"your app's config failed to load: {exc}",
            {"missing_env": _extract_missing_env(exc), "failing_module": mod_name},
        ) from exc

    if isinstance(exc, ModuleNotFoundError):
        raise ExtractError(
            "import_error",
            f"module '{mod_name}' could not be imported: {exc}",
            {"failing_module": mod_name},
        ) from exc

    raise ExtractError(
        "unknown",
        f"unexpected error importing '{mod_name}': {exc}",
        {"failing_module": mod_name, "traceback": traceback.format_exc()},
    ) from exc


# --- factory resolution: async context managers + checkpointer injection ---
#
# Common production pattern this handles:
#   def _compile_x_graph(checkpointer):        # private, does the real work
#       ...
#       return builder.compile()
#   async def x_graph():                       # @asynccontextmanager, prod entry point
#       async with ...: yield _compile_x_graph(real_checkpointer)
#   def build_x_graph(checkpointer=None):       # sync test/dev helper
#       return _compile_x_graph(checkpointer or InMemorySaver())
#
# Key fact this relies on: checkpointer CHOICE never affects graph structure
# (nodes/edges/routing are identical regardless of checkpointer backend), so
# substituting an in-memory one is always safe for visualization -- even
# though it would be wrong for production use. Every substitution is logged;
# never silent.
_CHECKPOINTER_NAME_HINTS = {"checkpointer", "saver", "checkpoint_saver", "checkpointsaver"}


def _is_asynccontextmanager_wrapper(obj):
    """If obj looks like a contextlib.asynccontextmanager-wrapped function
    (the `@asynccontextmanager async def x(): ... yield ...` pattern),
    return the original wrapped async-generator function -- functools.wraps
    (which asynccontextmanager uses internally) sets __wrapped__ to it.
    Purely introspective: never calls, awaits, or enters obj."""
    wrapped = getattr(obj, "__wrapped__", None)
    if wrapped is not None and inspect.isasyncgenfunction(wrapped):
        return wrapped
    return None


def _find_wrapped_call_name(fn) -> str | None:
    """AST-only: the name of a function called directly in a `yield` or
    `return` statement inside fn's body (e.g. `yield _compile_x(cp)` inside
    `async with ...:`). Never executes fn. First match wins."""
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return None
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Yield, ast.Return)) and isinstance(node.value, ast.Call):
            call_func = node.value.func
            if isinstance(call_func, ast.Name):
                return call_func.id
    return None


def _looks_like_checkpointer_param(name: str, annotation) -> bool:
    if name.lower() in _CHECKPOINTER_NAME_HINTS:
        return True
    if annotation is inspect.Parameter.empty or annotation is None:
        return False
    try:
        from langgraph.checkpoint.base import BaseCheckpointSaver

        if isinstance(annotation, type) and issubclass(annotation, BaseCheckpointSaver):
            return True
    except Exception:
        pass
    return "checkpoint" in str(annotation).lower()


def _make_in_memory_saver():
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


def _required_params(fn) -> list[tuple[str, object]]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return []
    return [
        (name, p.annotation)
        for name, p in sig.parameters.items()
        if p.default is inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]


def _build_shim_suggestion(module_name: str, fn_name: str, resolved: dict, remaining: list[str]) -> str:
    call_args = [f"{name}=InMemorySaver()" for name in resolved] + [f"<{name}>" for name in remaining]
    viz_name = fn_name.lstrip("_") or "graph"
    shim_path = f"scripts/_viz_{viz_name}.py"
    lines = ["retry with an explicit shim, e.g.:", f"  {shim_path}:"]
    if resolved:
        lines.append("    from langgraph.checkpoint.memory import InMemorySaver")
    lines.append(f"    from {module_name} import {fn_name}")
    lines.append(f"    graph = {fn_name}({', '.join(call_args)})")
    lines.append(f"  then: render-langgraph {shim_path}:graph")
    return "\n".join(lines)


def _resolve_callable(mod, attr: str, obj, spec: str, module_name: str):
    current = obj
    current_name = attr
    resolution_path: list[str] = []

    # Step 2: unwrap async context managers (possibly chained -- an async CM
    # could wrap another async CM in principle, so loop with a defensive cap
    # rather than assuming exactly one level).
    for _ in range(5):
        wrapped_gen_fn = _is_asynccontextmanager_wrapper(current)
        if wrapped_gen_fn is None:
            break
        _log(f"'{current_name}' is an async context manager; can't execute directly -- inspecting its source for the wrapped compile call")
        inner_name = _find_wrapped_call_name(wrapped_gen_fn)
        if inner_name is None:
            raise ExtractError(
                "unknown",
                f"'{current_name}' is an async context manager and its wrapped compile call couldn't be found statically",
                {"callable": current_name},
            )
        inner_obj = getattr(mod, inner_name, None)
        if inner_obj is None or not callable(inner_obj):
            raise ExtractError(
                "unknown",
                f"'{current_name}' wraps '{inner_name}', but that isn't a callable in this module",
                {"callable": current_name},
            )
        _log(f"found {inner_name}(...) inside {current_name}() -- resolving that instead")
        resolution_path.append("async_cm_unwrapped")
        current = inner_obj
        current_name = inner_name

    # Step 3: checkpointer auto-injection, then call.
    required = _required_params(current)
    kwargs: dict = {}
    checkpointer_arg = [name for name, ann in required if _looks_like_checkpointer_param(name, ann)]
    if len(checkpointer_arg) == 1:
        cp_name = checkpointer_arg[0]
        try:
            kwargs[cp_name] = _make_in_memory_saver()
            _log(
                f"{current_name} requires '{cp_name}'; this doesn't affect graph structure -- "
                f"substituting InMemorySaver() for visualization only"
            )
            resolution_path.append("checkpointer_injected")
        except Exception:
            pass  # couldn't build one; falls through to factory_needs_args below

    remaining = [name for name, _ in required if name not in kwargs]
    if remaining:
        shim = _build_shim_suggestion(module_name, current_name, kwargs, remaining)
        resolved_note = " (checkpointer already resolved automatically)" if kwargs else ""
        raise ExtractError(
            "factory_needs_args",
            f"'{current_name}' still requires: {', '.join(remaining)}{resolved_note}\n{shim}",
            {"callable": current_name, "unresolved_args": remaining, "resolved_args": list(kwargs)},
        )

    try:
        result = current(**kwargs)
        if resolution_path:
            _log(f"resolved via: {' -> '.join(resolution_path)}")
        else:
            _log(f"factory called: {current_name}()")
    except TypeError as exc:
        sig_msg = str(exc)
        if "argument" in sig_msg or "positional" in sig_msg:
            raise ExtractError(
                "factory_needs_args",
                f"'{current_name}' is a factory that requires arguments: {exc}",
                {"callable": current_name},
            ) from exc
        raise ExtractError(
            "unknown",
            f"calling '{current_name}' failed: {exc}",
            {"callable": current_name, "traceback": traceback.format_exc()},
        ) from exc
    except Exception as exc:
        if _looks_config_related(exc):
            raise ExtractError(
                "config_error",
                f"your app's config failed to load: {exc}",
                {"missing_env": _extract_missing_env(exc), "callable": current_name},
            ) from exc
        raise ExtractError(
            "unknown",
            f"calling '{current_name}' raised: {exc}",
            {"callable": current_name, "traceback": traceback.format_exc()},
        ) from exc

    if hasattr(result, "get_graph"):
        return result
    raise ExtractError(
        "not_a_graph",
        f"'{current_name}()' did not return a compiled graph",
        {"callable": current_name},
    )


def _resolve_attr(mod, attr: str, spec: str, module_name: str):
    obj = getattr(mod, attr, _MISSING)
    if obj is _MISSING:
        candidates = []
        for name, value in vars(mod).items():
            if name.startswith("_"):
                continue
            if hasattr(value, "get_graph") or type(value).__name__ == "StateGraph":
                candidates.append(name)
        raise ExtractError(
            "attr_missing",
            f"module has no attribute '{attr}'",
            {"available": candidates, "spec": spec},
        )

    if hasattr(obj, "get_graph"):
        return obj

    if type(obj).__name__ == "StateGraph":
        raise ExtractError(
            "not_a_graph",
            "that's the uncompiled builder -- point at the compiled graph, not the StateGraph",
            {"spec": spec},
        )

    if callable(obj):
        return _resolve_callable(mod, attr, obj, spec, module_name)

    raise ExtractError(
        "not_a_graph",
        f"'{attr}' is not a compiled graph, a factory, or a StateGraph builder",
        {"spec": spec},
    )


def _unwrap_callable(data):
    for attr_name in ("func", "afunc", "__wrapped__"):
        inner = getattr(data, attr_name, None)
        if inner is not None and callable(inner):
            return inner
    return data if callable(data) else None


def _source_location(data):
    fn = _unwrap_callable(data)
    if fn is None:
        return None, None
    try:
        file = inspect.getsourcefile(fn)
        _, line = inspect.getsourcelines(fn)
        return (os.path.abspath(file) if file else None), line
    except (TypeError, OSError):
        return None, None


_SUBGRAPH_TYPE_NAMES = {"Pregel", "CompiledStateGraph", "CompiledGraph"}


def _node_kind(node_id: str, data):
    if node_id in _VIRTUAL_NODES:
        return _VIRTUAL_NODES[node_id]
    if data is None:
        return "runnable"
    type_name = type(data).__name__
    if "ToolNode" in type_name:
        return "tool"
    if type_name in _SUBGRAPH_TYPE_NAMES:
        return "subgraph"
    fn = _unwrap_callable(data)
    if fn is not None:
        return "function"
    return "runnable"


def _node_namespace(node_id: str) -> str | None:
    if ":" in node_id:
        return node_id.rsplit(":", 1)[0]
    return None


def _literal_values(annotation) -> list[str] | None:
    if annotation is None:
        return None
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        return [str(a) for a in typing.get_args(annotation)]
    if origin is typing.Union:
        values: list[str] = []
        for arg in typing.get_args(annotation):
            sub = _literal_values(arg)
            if sub is None:
                return None
            values.extend(sub)
        return values
    return None


# Branch classification -- render-langgraph's signature feature: for each
# router (the callable passed to add_conditional_edges), statically decide
# whether it's a real (LLM-driven / open) decision or a disguised
# deterministic if-else. Static only: read source with ast, never execute
# the router. Kept as one conservative, centralized allowlist so it's easy
# to extend and so a false "dynamic" (worse than "unknown") stays unlikely.
_MODEL_LIKE_CLASS_NAMES = {
    "ChatOpenAI", "AzureChatOpenAI", "ChatAnthropic", "ChatVertexAI",
    "ChatGoogleGenerativeAI", "ChatCohere", "ChatMistralAI", "ChatOllama",
    "ChatFireworks", "ChatGroq", "ChatBedrock", "ChatBedrockConverse",
    "ChatHuggingFace", "ChatLiteLLM", "OpenAI", "Anthropic", "Ollama",
}
_MODEL_CALL_METHODS = {"invoke", "ainvoke", "stream", "astream", "batch", "abatch"}
_MODEL_BIND_METHODS = {"bind_tools", "with_structured_output"}


def _describe_call_target(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return "<expr>"


def _find_model_like_call(tree) -> str | None:
    """Return a short description of the first model-like call found in an
    AST subtree, or None. Only matches signals strong enough on their own:
    known chat-model constructors, or the .invoke-family of methods (which
    in a router body overwhelmingly means "call a runnable/chain/llm", not
    some unrelated object)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in _MODEL_LIKE_CLASS_NAMES:
            return f"constructs {func.id}(...)"
        if isinstance(func, ast.Attribute):
            if func.attr in _MODEL_LIKE_CLASS_NAMES:
                return f"constructs .{func.attr}(...)"
            if func.attr in _MODEL_CALL_METHODS or func.attr in _MODEL_BIND_METHODS:
                target = _describe_call_target(func.value)
                return f"calls {target}.{func.attr}(...)"
    return None


def _classify_router(router_fn, has_fixed_targets: bool) -> tuple[str, str]:
    """Static classification of a router callable. Never imports/executes
    it -- reads inspect.getsource() + ast only. Anything that can't be
    confidently resolved is "unknown", never guessed as "dynamic"."""
    if router_fn is None:
        return "unknown", "no router callable resolved"

    if getattr(router_fn, "__name__", None) == "<lambda>":
        return "unknown", "router is a lambda; source too limited to classify safely"

    try:
        source = inspect.getsource(router_fn)
    except (OSError, TypeError):
        return "unknown", "router source not resolvable"

    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return "unknown", "router source not parseable"

    func_node = next(
        (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if func_node is None:
        return "unknown", "router source has no inspectable function body"

    model_call = _find_model_like_call(func_node)
    if model_call:
        return "dynamic", f"{model_call} in router"

    if has_fixed_targets:
        return "deterministic", "fixed target set (Literal/path_map), no model call in router"

    return "unknown", "no model call detected but no fixed target set either"


def _analyze_branches(compiled) -> tuple[dict, dict]:
    """Single pass over compiled.builder.branches producing:
      - edge_overrides: (source, target) -> {"label", "inferred"}
      - router_classifications: source -> {"decision_kind", "decision_reason"}
    Kept together (rather than a second pass) because path_map/Literal
    resolution is already needed for edge narrowing -- classification reuses
    exactly that, it doesn't re-derive it. Falls back silently (empty dicts)
    across LangGraph internal-API changes.
    """
    edge_overrides: dict = {}
    router_classifications: dict = {}
    try:
        branches = compiled.builder.branches
    except AttributeError:
        return edge_overrides, router_classifications

    for source, branch_map in branches.items():
        for branch in branch_map.values():
            path_map = getattr(branch, "ends", None)
            router = getattr(branch, "path", None)
            router_fn = _unwrap_callable(router) or router

            literal_targets = None
            if router_fn is not None:
                try:
                    hints = typing.get_type_hints(router_fn)
                    literal_targets = _literal_values(hints.get("return"))
                except Exception:
                    literal_targets = None

            has_fixed_path_map = isinstance(path_map, dict) and bool(path_map)
            has_fixed_targets = has_fixed_path_map or bool(literal_targets)

            if has_fixed_path_map:
                for label, target in path_map.items():
                    edge_overrides[(source, target)] = {"label": str(label), "inferred": False}
            elif literal_targets:
                for target in literal_targets:
                    edge_overrides[(source, target)] = {"label": None, "inferred": True}

            kind, reason = _classify_router(router_fn, has_fixed_targets)
            router_classifications[source] = {"decision_kind": kind, "decision_reason": reason}

    return edge_overrides, router_classifications


def _flatten(compiled, xray: bool) -> dict:
    g = compiled.get_graph(xray=xray)
    overrides, router_classifications = _analyze_branches(compiled)

    nodes = []
    for node_id, node in g.nodes.items():
        file, line = _source_location(getattr(node, "data", None))
        classification = router_classifications.get(node_id)
        nodes.append(
            {
                "id": node_id,
                "kind": _node_kind(node_id, getattr(node, "data", None)),
                "file": file,
                "line": line,
                "subgraph": _node_namespace(node_id),
                "decision_kind": classification["decision_kind"] if classification else None,
                "decision_reason": classification["decision_reason"] if classification else None,
            }
        )

    edges = []
    for edge in g.edges:
        key = (edge.source, edge.target)
        override = overrides.get(key)
        conditional = bool(getattr(edge, "conditional", False))
        label = getattr(edge, "data", None)
        inferred = False
        branch_kind = None
        if conditional:
            if override is not None:
                label = override["label"] if override["label"] is not None else label
                inferred = override["inferred"]
            else:
                inferred = label is None
            source_classification = router_classifications.get(edge.source)
            branch_kind = source_classification["decision_kind"] if source_classification else None
        edges.append(
            {
                "src": edge.source,
                "tgt": edge.target,
                "conditional": conditional,
                "label": label,
                "inferred": inferred,
                "branch_kind": branch_kind,
            }
        )

    topology = json.dumps(
        {
            "nodes": sorted((n["id"], n["kind"], n["decision_kind"]) for n in nodes),
            "edges": sorted((e["src"], e["tgt"], e["conditional"], e["inferred"], e["branch_kind"]) for e in edges),
        },
        sort_keys=True,
    )
    graph_hash = hashlib.sha256(topology.encode("utf-8")).hexdigest()

    return {
        "graph_id": getattr(compiled, "name", None) or "graph",
        "hash": graph_hash,
        "nodes": nodes,
        "edges": edges,
        "subgraphs": [],
    }


def run(project_root: str, module_name: str, attr: str, xray: bool) -> dict:
    dotenv_loaded = [False]

    mod = _import_target(project_root, module_name, dotenv_loaded)
    compiled = _resolve_attr(mod, attr, f"{module_name}:{attr}", module_name)
    return _flatten(compiled, xray)


def main() -> None:
    project_root, module_name, attr, output_file, xray_flag = sys.argv[1:6]
    xray = xray_flag == "1"
    try:
        result = run(project_root, module_name, attr, xray)
    except ExtractError as exc:
        result = {
            "error": exc.message,
            "kind": exc.kind,
            "detail": exc.detail,
            "traceback": exc.detail.get("traceback", traceback.format_exc()),
        }
    except Exception as exc:  # backstop: anything unanticipated
        result = {
            "error": str(exc),
            "kind": "unknown",
            "detail": {},
            "traceback": traceback.format_exc(),
        }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
