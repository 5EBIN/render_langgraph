"""Each predictable failure kind from HANDOFF.md's table, plus the
invariant that matters most: an import failure never silently becomes an
AST render (there is no fallback path from extract() to static_parse()
anywhere in the codebase -- --static is opt-in only, checked here by
asserting the failure results carry no AST-shaped fields)."""
from render_langgraph.extract import extract


def test_factory_needs_args(target_python, target_project):
    result = extract(target_python, "factory.py:build_graph", target_project)
    assert not result.ok
    assert result.data["kind"] == "factory_needs_args"
    assert result.data["detail"]["callable"] == "build_graph"


def test_attr_missing_lists_available_graphs(target_python, target_project):
    result = extract(target_python, "wrong_attr.py:graph", target_project)
    assert not result.ok
    assert result.data["kind"] == "attr_missing"
    assert "compiled_graph" in result.data["detail"]["available"]


def test_not_a_graph_points_at_builder(target_python, target_project):
    result = extract(target_python, "wrong_attr.py:builder", target_project)
    assert not result.ok
    assert result.data["kind"] == "not_a_graph"
    assert "builder" in result.data["error"]


def test_import_error_names_the_module(target_python, target_project):
    result = extract(target_python, "broken_import.py:graph", target_project)
    assert not result.ok
    assert result.data["kind"] == "import_error"
    assert result.data["detail"]["failing_module"] == "broken_import"


def test_config_error_names_missing_env(target_python, target_project):
    result = extract(target_python, "config_fail.py:graph", target_project)
    assert not result.ok
    assert result.data["kind"] == "config_error"
    assert "DATABASE_URL" in result.data["detail"]["missing_env"]


def test_config_error_self_fixes_via_dotenv(target_python, target_project, tmp_path, monkeypatch):
    dotenv = target_project / ".env"
    dotenv.write_text("DATABASE_URL=postgres://example\n", encoding="utf-8")
    try:
        result = extract(target_python, "config_fail.py:graph", target_project)
        assert result.ok, result.data
    finally:
        dotenv.unlink()


def test_timeout_is_classified_not_hung(target_python, target_project):
    result = extract(target_python, "slow_import.py:graph", target_project, hard_timeout=3, soft_timeout=1)
    assert not result.ok
    assert result.data["kind"] == "timeout"


def test_failures_never_carry_ast_shaped_fields(target_python, target_project):
    """The invariant: a runtime failure result is never quietly replaced by
    a static_parse()-style success payload. Failure results have 'error'/'kind'
    and no 'nodes'/'edges' keys with real content."""
    for spec in ("factory.py:build_graph", "wrong_attr.py:graph", "broken_import.py:graph"):
        result = extract(target_python, spec, target_project)
        assert not result.ok
        assert "error" in result.data and "kind" in result.data
        assert "nodes" not in result.data
        assert "edges" not in result.data
