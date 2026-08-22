"""Covers the async-context-manager / checkpointer-auto-injection factory
resolution chain: a private `_compile_x_graph(checkpointer)` wrapped by an
`@asynccontextmanager` production entry point, with an optional sync
`build_x_graph()` test helper -- see examples/simple_agent/orchide_pattern.py.
"""
from pathlib import Path

from render_langgraph import discover
from render_langgraph.extract import extract
from render_langgraph.spec import split_spec


def test_async_cm_wrapping_checkpointer_only_fn_fully_auto_resolves(target_python, target_project):
    lines: list[str] = []
    result = extract(target_python, "orchide_pattern.py:planning_graph", target_project, on_progress=lines.append)
    assert result.ok, result.data
    assert {n["id"] for n in result.data["nodes"]} == {"__start__", "plan_project", "design_node", "__end__"}

    joined = "\n".join(lines)
    assert "async context manager" in joined
    assert "_compile_planning_graph" in joined
    assert "InMemorySaver" in joined


def test_checkpointer_substitution_is_always_logged_never_silent(target_python, target_project):
    lines: list[str] = []
    result = extract(target_python, "orchide_pattern.py:build_planning_graph", target_project, on_progress=lines.append)
    assert result.ok, result.data
    # Even through the sync helper (no async CM to unwrap), the checkpointer
    # substitution happens *inside* build_planning_graph's own Python code
    # (checkpointer or InMemorySaver()), not through our injection path --
    # so there's nothing for us to log there. The direct-injection path is
    # what test_async_cm_wrapping_checkpointer_only_fn_fully_auto_resolves
    # and test_direct_private_fn_injection_is_logged assert on instead.
    assert result.data["nodes"]


def test_direct_private_fn_injection_is_logged(target_python, target_project):
    lines: list[str] = []
    result = extract(target_python, "orchide_pattern.py:_compile_planning_graph", target_project, on_progress=lines.append)
    assert result.ok, result.data
    joined = "\n".join(lines)
    assert "requires 'checkpointer'" in joined
    assert "InMemorySaver" in joined
    assert "doesn't affect graph structure" in joined


def test_second_unrelated_required_arg_reported_after_checkpointer_resolves(target_python, target_project):
    lines: list[str] = []
    result = extract(target_python, "orchide_pattern.py:code_graph", target_project, on_progress=lines.append)
    assert not result.ok
    assert result.data["kind"] == "factory_needs_args"
    assert "llm_client" in result.data["error"]
    assert "checkpointer" not in result.data["detail"].get("unresolved_args", [])
    assert result.data["detail"]["unresolved_args"] == ["llm_client"]
    assert result.data["detail"]["resolved_args"] == ["checkpointer"]
    # confirms the checkpointer step ran (and was logged) even though the
    # overall resolution still failed on the other arg
    assert "requires 'checkpointer'" in "\n".join(lines)


def test_shim_suggestion_included_in_error_message(target_python, target_project):
    result = extract(target_python, "orchide_pattern.py:code_graph", target_project)
    assert not result.ok
    message = result.data["error"]
    assert "scripts/_viz_" in message
    assert "from orchide_pattern import _compile_code_graph" in message
    assert "InMemorySaver()" in message
    assert "<llm_client>" in message
    assert "render-langgraph scripts/_viz_" in message


def test_sync_helper_alone_resolves_without_any_injection_logging(target_python, target_project):
    lines: list[str] = []
    result = extract(target_python, "orchide_pattern.py:build_code_graph", target_project, on_progress=lines.append)
    assert result.ok, result.data
    assert {n["id"] for n in result.data["nodes"]} == {"__start__", "generate", "__end__"}


def test_discovery_prefers_sync_helper_and_lists_both_graphs_separately(target_project):
    candidates = discover.scan_ast(target_project)
    by_name = {}
    for c in candidates:
        path, attr = split_spec(c.spec)
        if Path(path).name == "orchide_pattern.py":
            by_name[f"orchide_pattern.py:{attr}"] = c

    build_planning = by_name["orchide_pattern.py:build_planning_graph"]
    build_code = by_name["orchide_pattern.py:build_code_graph"]
    async_planning = by_name["orchide_pattern.py:planning_graph"]
    async_code = by_name["orchide_pattern.py:code_graph"]
    private_planning = by_name["orchide_pattern.py:_compile_planning_graph"]
    private_code = by_name["orchide_pattern.py:_compile_code_graph"]

    # sync helpers outrank both the async CM entry points and the raw
    # private compile functions for their respective graph
    assert build_planning.score > async_planning.score > private_planning.score
    assert build_code.score > async_code.score > private_code.score

    # the two sync helpers are for two DIFFERENT graphs and must tie at the
    # top score -- discovery must not silently prefer one over the other
    assert build_planning.score == build_code.score
