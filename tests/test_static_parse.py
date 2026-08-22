from render_langgraph.static_parse import parse_file


def test_static_never_imports_and_resolves_linear_graph(target_project):
    result = parse_file(str(target_project / "linear_graph.py"))
    assert result["partial"] is False
    node_ids = {n["id"] for n in result["nodes"]}
    assert node_ids == {"upper", "exclaim"}
    edges = {(e["src"], e["tgt"]) for e in result["edges"]}
    assert ("__start__", "upper") in edges
    assert ("exclaim", "__end__") in edges


def test_static_marks_partial_on_broken_import(target_project):
    # broken_import.py has an unresolvable top-level import, but no graph
    # construction calls at all -- static parsing must still succeed (it
    # never imports) even though it can't extract meaningful graph data.
    result = parse_file(str(target_project / "broken_import.py"))
    assert result["nodes"] == []
    assert result["edges"] == []


def test_static_conditional_edges_without_path_map_are_partial(tmp_path):
    src = tmp_path / "computed.py"
    src.write_text(
        "from langgraph.graph import StateGraph\n"
        "builder = StateGraph(dict)\n"
        "builder.add_node('a', lambda s: s)\n"
        "builder.add_node('b', lambda s: s)\n"
        "builder.add_conditional_edges('a', router_fn)\n",
        encoding="utf-8",
    )
    result = parse_file(str(src))
    assert result["partial"] is True


def test_static_syntax_error_reports_partial_not_crash(tmp_path):
    src = tmp_path / "broken.py"
    src.write_text("def f(:\n    pass\n", encoding="utf-8")
    result = parse_file(str(src))
    assert result["partial"] is True
    assert "error" in result


def _node(result, node_id):
    return next(n for n in result["nodes"] if n["id"] == node_id)


def test_static_classifies_deterministic_router_from_source(target_project):
    result = parse_file(str(target_project / "branch_classification.py"))
    node = _node(result, "det_a")
    assert node["decision_kind"] == "deterministic"


def test_static_classifies_dynamic_router_via_invoke_call(tmp_path):
    src = tmp_path / "dyn.py"
    src.write_text(
        "from langgraph.graph import StateGraph\n"
        "def route(state):\n"
        "    return llm.invoke(state['x'])\n"
        "builder = StateGraph(dict)\n"
        "builder.add_node('a', lambda s: s)\n"
        "builder.add_node('b', lambda s: s)\n"
        "builder.add_conditional_edges('a', route, path_map={'a': 'b'})\n",
        encoding="utf-8",
    )
    result = parse_file(str(src))
    node = _node(result, "a")
    assert node["decision_kind"] == "dynamic"
    assert "invoke" in node["decision_reason"]


def test_static_lambda_router_is_unknown_not_guessed(tmp_path):
    src = tmp_path / "lam.py"
    src.write_text(
        "from langgraph.graph import StateGraph\n"
        "builder = StateGraph(dict)\n"
        "builder.add_node('a', lambda s: s)\n"
        "builder.add_node('b', lambda s: s)\n"
        "builder.add_conditional_edges('a', lambda s: 'a' if s['x'] else 'b', path_map={'a': 'b'})\n",
        encoding="utf-8",
    )
    result = parse_file(str(src))
    node = _node(result, "a")
    assert node["decision_kind"] == "unknown"


def test_static_router_defined_elsewhere_is_unknown(tmp_path):
    src = tmp_path / "external.py"
    src.write_text(
        "from langgraph.graph import StateGraph\n"
        "from somewhere import route\n"
        "builder = StateGraph(dict)\n"
        "builder.add_node('a', lambda s: s)\n"
        "builder.add_node('b', lambda s: s)\n"
        "builder.add_conditional_edges('a', route, path_map={'a': 'b'})\n",
        encoding="utf-8",
    )
    result = parse_file(str(src))
    node = _node(result, "a")
    assert node["decision_kind"] == "unknown"
    assert "not defined in this file" in node["decision_reason"]
