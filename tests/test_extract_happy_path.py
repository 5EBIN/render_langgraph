from render_langgraph.extract import extract


def test_linear_graph(target_python, target_project):
    result = extract(target_python, "linear_graph.py:graph", target_project)
    assert result.ok, result.data
    node_ids = {n["id"] for n in result.data["nodes"]}
    assert node_ids == {"__start__", "upper", "exclaim", "__end__"}
    edges = {(e["src"], e["tgt"]) for e in result.data["edges"]}
    assert edges == {("__start__", "upper"), ("upper", "exclaim"), ("exclaim", "__end__")}
    assert all(not e["conditional"] for e in result.data["edges"])


def test_agent_tools_conditional_edges_are_narrowed(target_python, target_project):
    result = extract(target_python, "agent_tools.py:graph", target_project)
    assert result.ok, result.data
    conditional = [e for e in result.data["edges"] if e["conditional"]]
    targets = {e["tgt"] for e in conditional}
    # The correctness win: only the two Literal-declared branches, not a fan-out
    # to every node in the graph.
    assert targets == {"tools", "__end__"}


def test_nested_subgraph_is_flattened_with_namespace(target_python, target_project):
    result = extract(target_python, "nested.py:graph", target_project)
    assert result.ok, result.data
    node_ids = {n["id"] for n in result.data["nodes"]}
    assert "researcher:sub_step" in node_ids
    sub_node = next(n for n in result.data["nodes"] if n["id"] == "researcher:sub_step")
    assert sub_node["subgraph"] == "researcher"
    assert sub_node["kind"] == "function"


def test_absolute_windows_path_spec_does_not_collide_with_drive_colon(target_python, target_project):
    # Regression: "C:/.../graph.py:graph" naively split on the first ":"
    # used to chop the spec at the drive letter instead of the attr suffix.
    abs_spec = f"{(target_project / 'linear_graph.py').as_posix()}:graph"
    result = extract(target_python, abs_spec, target_project)
    assert result.ok, result.data
    assert {n["id"] for n in result.data["nodes"]} == {"__start__", "upper", "exclaim", "__end__"}


def test_hash_is_stable_across_runs(target_python, target_project):
    a = extract(target_python, "linear_graph.py:graph", target_project)
    b = extract(target_python, "linear_graph.py:graph", target_project)
    assert a.data["hash"] == b.data["hash"]
    assert a.data["hash"] is not None


def test_garbage_stdout_at_import_does_not_corrupt_json_result(target_python, target_project):
    # The extractor hands back its result via a temp file, not stdout --
    # this proves it: the target module prints non-JSON garbage at import
    # and mid-execution, and the result must still parse cleanly.
    result = extract(target_python, "stdout_garbage.py:graph", target_project)
    assert result.ok, result.data
    assert {n["id"] for n in result.data["nodes"]} == {"__start__", "step", "__end__"}


def test_zero_arg_factory_is_found_called_and_extracted(target_python, target_project):
    result = extract(target_python, "factory_ok.py:build_graph", target_project)
    assert result.ok, result.data
    assert {n["id"] for n in result.data["nodes"]} == {"__start__", "step", "__end__"}
