import json

from render_langgraph import discover


def test_explicit_target_bypasses_discovery(tmp_path):
    result = discover.resolve(explicit="a/b.py:graph", project_root=tmp_path)
    assert result.resolved.spec == "a/b.py:graph"
    assert result.resolved.source == "explicit"


def test_langgraph_json_single_entry_resolves(tmp_path):
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "graph.py").write_text("graph = None\n", encoding="utf-8")
    (tmp_path / "langgraph.json").write_text(
        json.dumps({"graphs": {"agent": "./agent/graph.py:graph"}}), encoding="utf-8"
    )
    result = discover.resolve(explicit=None, project_root=tmp_path, use_cache=False)
    assert result.resolved is not None
    assert result.resolved.spec.endswith("agent/graph.py:graph")


def test_langgraph_json_multiple_entries_needs_picker_or_name(tmp_path):
    (tmp_path / "langgraph.json").write_text(
        json.dumps({"graphs": {"a": "./a.py:graph", "b": "./b.py:graph"}}), encoding="utf-8"
    )
    result = discover.resolve(explicit=None, project_root=tmp_path, use_cache=False)
    assert result.resolved is None
    assert len(result.candidates) == 2

    named = discover.resolve(explicit=None, project_root=tmp_path, graph_name="b", use_cache=False)
    assert named.resolved.spec.endswith("b.py:graph")


def test_ast_scan_finds_compiled_graph_and_factory(target_project):
    candidates = discover.scan_ast(target_project)
    specs = {c.spec.split("/")[-1]: c for c in candidates}
    assert "linear_graph.py:graph" in specs
    assert specs["linear_graph.py:graph"].kind == "compiled"

    factory = specs["factory.py:build_graph"]
    assert factory.kind == "factory"
    assert factory.needs_args is True


def test_cache_roundtrip_and_gitignore(tmp_path):
    candidate = discover.Candidate(spec="x.py:graph", source="ast", kind="compiled")
    discover.save_cache(tmp_path, candidate)

    loaded = discover.load_cache(tmp_path)
    assert loaded.spec == "x.py:graph"

    gitignore = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert ".render-langgraph/" in gitignore
