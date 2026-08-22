"""AST-only fallback (`--static`). Never imports the target.

Handles unsaved/broken files. Cannot resolve anything computed
(e.g. `for name in cfg: builder.add_node(name, fn)`) -- that's inherent,
not a bug, so callers get "partial": true when we hit constructs we can't
statically resolve.
"""
from __future__ import annotations

import ast
from pathlib import Path

_GRAPH_METHODS = {
    "add_node", "add_edge", "add_conditional_edges",
    "set_entry_point", "set_finish_point", "add_sequence",
}

# Branch classification allowlist -- intentionally duplicated from
# _extractor.py rather than shared. _extractor.py must stay a standalone
# script importable with zero render_langgraph dependencies (it runs in the
# TARGET's venv); this module runs in the tool's own interpreter and could
# import a shared constant, but keeping both copies small and local matches
# the precedent already set for spec.py/_extractor.py's split_spec logic.
_MODEL_LIKE_CLASS_NAMES = {
    "ChatOpenAI", "AzureChatOpenAI", "ChatAnthropic", "ChatVertexAI",
    "ChatGoogleGenerativeAI", "ChatCohere", "ChatMistralAI", "ChatOllama",
    "ChatFireworks", "ChatGroq", "ChatBedrock", "ChatBedrockConverse",
    "ChatHuggingFace", "ChatLiteLLM", "OpenAI", "Anthropic", "Ollama",
}
_MODEL_CALL_METHODS = {"invoke", "ainvoke", "stream", "astream", "batch", "abatch"}
_MODEL_BIND_METHODS = {"bind_tools", "with_structured_output"}


def _literal_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    return None


_WELL_KNOWN_NAMES = {"START": "__start__", "END": "__end__"}


def _node_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return _WELL_KNOWN_NAMES.get(node.id, f"<{node.id}>")
    if isinstance(node, ast.Attribute):
        return _WELL_KNOWN_NAMES.get(node.attr, f"<{node.attr}>")
    return None


def _describe_call_target(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return "<expr>"


def _find_model_like_call(tree: ast.AST) -> str | None:
    """Same conservative signal set as _extractor.py: known chat-model
    constructors, or the .invoke-family of methods."""
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


def _find_function_def(tree: ast.AST, name: str):
    return next(
        (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name),
        None,
    )


def _is_literal_annotation(node: ast.AST) -> bool:
    if isinstance(node, ast.Subscript):
        value = node.value
        name = value.id if isinstance(value, ast.Name) else (value.attr if isinstance(value, ast.Attribute) else None)
        return name == "Literal"
    return False


def _classify_router_static(tree: ast.AST, router_arg: ast.AST | None, has_fixed_targets: bool) -> tuple[str, str]:
    """Best-effort static classification without ever resolving/importing
    the router. Lands on "unknown" far more often than the runtime path in
    _extractor.py -- that's expected and correct, not a regression; static
    mode has no import, so it can't chase the callable across modules."""
    if router_arg is None:
        return "unknown", "router argument not found"

    if isinstance(router_arg, ast.Lambda):
        model_call = _find_model_like_call(router_arg)
        if model_call:
            return "dynamic", f"{model_call} in router"
        return "unknown", "router is a lambda; source too limited to classify safely"

    if not isinstance(router_arg, ast.Name):
        return "unknown", "router is not a simple named function reference"

    func_def = _find_function_def(tree, router_arg.id)
    if func_def is None:
        return "unknown", "router function not defined in this file"

    model_call = _find_model_like_call(func_def)
    if model_call:
        return "dynamic", f"{model_call} in router"

    if has_fixed_targets:
        return "deterministic", "fixed target set (path_map/Literal), no model call in router"

    return "unknown", "no model call detected but no fixed target set either"


def parse_file(path: str) -> dict:
    file_path = Path(path)
    source = file_path.read_text(encoding="utf-8")
    partial = False

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        return {
            "graph_id": file_path.stem,
            "hash": None,
            "nodes": [],
            "edges": [],
            "subgraphs": [],
            "partial": True,
            "error": f"syntax error at line {exc.lineno}: {exc.msg}",
        }

    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    entry: str | None = None
    router_classifications: dict[str, tuple[str, str]] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr not in _GRAPH_METHODS:
            continue
        method = node.func.attr
        args = node.args
        line = node.lineno
        col = node.col_offset

        if method == "add_node":
            if not args:
                partial = True
                continue
            name = _node_name(args[0])
            if name is None:
                partial = True
                continue
            nodes[name] = {
                "id": name, "kind": "function", "file": str(file_path.resolve()), "line": line,
                "subgraph": None, "decision_kind": None, "decision_reason": None,
            }

        elif method == "add_edge":
            if len(args) < 2:
                partial = True
                continue
            src, tgt = _node_name(args[0]), _node_name(args[1])
            if src is None or tgt is None:
                partial = True
                continue
            edges.append({"src": src, "tgt": tgt, "conditional": False, "label": None, "inferred": False, "branch_kind": None})

        elif method == "add_conditional_edges":
            if not args:
                partial = True
                continue
            src = _node_name(args[0])
            if src is None:
                partial = True
                continue
            path_map = None
            for kw in node.keywords:
                if kw.arg == "path_map" and isinstance(kw.value, ast.Dict):
                    path_map = kw.value
            if path_map is None and len(args) >= 3 and isinstance(args[2], ast.Dict):
                path_map = args[2]

            has_fixed_targets = path_map is not None
            router_arg = args[1] if len(args) >= 2 else None
            if not has_fixed_targets and isinstance(router_arg, ast.Name):
                func_def = _find_function_def(tree, router_arg.id)
                if func_def is not None and func_def.returns is not None:
                    has_fixed_targets = _is_literal_annotation(func_def.returns)

            decision_kind, decision_reason = _classify_router_static(tree, router_arg, has_fixed_targets)
            router_classifications[src] = (decision_kind, decision_reason)

            if path_map is not None:
                for k, v in zip(path_map.keys, path_map.values):
                    label = _literal_str(k)
                    tgt = _node_name(v)
                    if tgt is None:
                        partial = True
                        continue
                    edges.append({
                        "src": src, "tgt": tgt, "conditional": True, "label": label,
                        "inferred": False, "branch_kind": decision_kind,
                    })
            else:
                partial = True  # branch targets can't be resolved statically without path_map

        elif method == "set_entry_point":
            if args:
                entry = _node_name(args[0])

        elif method == "set_finish_point":
            if args:
                tgt = _node_name(args[0])
                if tgt:
                    edges.append({"src": tgt, "tgt": "__end__", "conditional": False, "label": None, "inferred": False, "branch_kind": None})

        elif method == "add_sequence":
            if args and isinstance(args[0], (ast.List, ast.Tuple)):
                seq_names = [_node_name(e) for e in args[0].elts]
                for i, name in enumerate(seq_names):
                    if name is None:
                        partial = True
                        continue
                    nodes[name] = {
                        "id": name, "kind": "function", "file": str(file_path.resolve()), "line": line,
                        "subgraph": None, "decision_kind": None, "decision_reason": None,
                    }
                    if i > 0 and seq_names[i - 1]:
                        edges.append({"src": seq_names[i - 1], "tgt": name, "conditional": False, "label": None, "inferred": False, "branch_kind": None})
            else:
                partial = True

    for src, (kind, reason) in router_classifications.items():
        if src in nodes:
            nodes[src]["decision_kind"] = kind
            nodes[src]["decision_reason"] = reason

    if entry:
        edges.append({"src": "__start__", "tgt": entry, "conditional": False, "label": None, "inferred": False, "branch_kind": None})
        nodes.setdefault("__start__", {
            "id": "__start__", "kind": "start", "file": None, "line": None,
            "subgraph": None, "decision_kind": None, "decision_reason": None,
        })

    return {
        "graph_id": file_path.stem,
        "hash": None,
        "nodes": list(nodes.values()),
        "edges": edges,
        "subgraphs": [],
        "partial": partial,
    }
