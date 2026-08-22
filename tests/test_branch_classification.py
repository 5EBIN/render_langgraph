"""render-langgraph's signature feature: for each router, statically decide
whether it's a real (LLM-driven) decision or a disguised deterministic
if-else. Covers the runtime path (_extractor.py, via extract()) for all 5
patterns, plus that existing edge-narrowing behavior is unaffected."""
from render_langgraph.extract import extract


def _node(result, node_id):
    return next(n for n in result.data["nodes"] if n["id"] == node_id)


def _conditional_edges(result):
    return [e for e in result.data["edges"] if e["conditional"]]


def test_deterministic_literal_return_no_model_call(target_python, target_project):
    result = extract(target_python, "branch_classification.py:graph_deterministic", target_project)
    assert result.ok, result.data
    node = _node(result, "det_a")
    assert node["decision_kind"] == "deterministic"
    assert "no model call" in node["decision_reason"]
    edges = _conditional_edges(result)
    assert edges, "expected at least one conditional edge from the router"
    assert all(e["branch_kind"] == "deterministic" for e in edges)


def test_dynamic_llm_invoke_call(target_python, target_project):
    result = extract(target_python, "branch_classification.py:graph_dynamic_llm", target_project)
    assert result.ok, result.data
    node = _node(result, "dyn_a")
    assert node["decision_kind"] == "dynamic"
    assert "invoke" in node["decision_reason"]
    edges = _conditional_edges(result)
    assert edges, "expected at least one conditional edge from the router"
    assert all(e["branch_kind"] == "dynamic" for e in edges)


def test_dynamic_no_hints_model_derived(target_python, target_project):
    result = extract(target_python, "branch_classification.py:graph_dynamic_no_hints", target_project)
    assert result.ok, result.data
    node = _node(result, "nohint_a")
    assert node["decision_kind"] == "dynamic"
    assert "invoke" in node["decision_reason"]
    # This langgraph version doesn't fan out conditional edges when a router
    # has neither path_map nor a Literal hint -- get_graph() simply can't
    # enumerate targets. The node classification is still correct; there's
    # just nothing on the edge side to propagate branch_kind onto here.


def test_unknown_dynamically_built_router(target_python, target_project):
    result = extract(target_python, "branch_classification.py:graph_unknown_dynamic", target_project)
    assert result.ok, result.data
    node = _node(result, "dynbuilt_a")
    assert node["decision_kind"] == "unknown"
    assert "not resolvable" in node["decision_reason"]


def test_unknown_lambda_router(target_python, target_project):
    result = extract(target_python, "branch_classification.py:graph_unknown_lambda", target_project)
    assert result.ok, result.data
    node = _node(result, "lam_a")
    assert node["decision_kind"] == "unknown"
    assert "lambda" in node["decision_reason"]
    # Lambdas can't carry a Literal[...] return annotation (no annotation
    # syntax for lambdas at all), and this graph has no path_map either --
    # so, same as the no-hints dynamic case, get_graph() has no fixed target
    # set to fan out over and produces zero conditional edges here. The node
    # classification is what matters and is asserted above.


def test_non_router_nodes_have_null_classification(target_python, target_project):
    result = extract(target_python, "branch_classification.py:graph_deterministic", target_project)
    assert result.ok, result.data
    for node_id in ("__start__", "det_b", "__end__"):
        node = _node(result, node_id)
        assert node["decision_kind"] is None
        assert node["decision_reason"] is None
    non_conditional = [e for e in result.data["edges"] if not e["conditional"]]
    assert non_conditional
    assert all(e["branch_kind"] is None for e in non_conditional)


def test_existing_edge_narrowing_unaffected(target_python, target_project):
    """Regression guard: the P1 correctness feature (no fan-out to every
    node when a router has path_map/Literal hints) must still work exactly
    as before -- classification is additive, not a replacement."""
    result = extract(target_python, "agent_tools.py:graph", target_project)
    assert result.ok, result.data
    conditional = [e for e in result.data["edges"] if e["conditional"]]
    targets = {e["tgt"] for e in conditional}
    assert targets == {"tools", "__end__"}
