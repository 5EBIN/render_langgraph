"""Regression: `.compile()` is too generic a method name to trust on its
own -- re.compile(...), a Jinja2 Environment.compile(), etc. all match the
same AST shape as a LangGraph builder.compile(). Found against a real
production module that had `_FENCE_OPEN = re.compile(...)` at module scope
and a plain helper function that called `re.compile()` internally; both
were misdetected as compiled-graph candidates before the StateGraph-bound-
name gate was added.
"""
from render_langgraph import discover


def test_module_level_re_compile_is_not_a_candidate(tmp_path):
    src = tmp_path / "nodes.py"
    src.write_text(
        "import re\n"
        "\n"
        "_FENCE_OPEN = re.compile(r'```\\w*\\n')\n"
        "_FENCE_CLOSE = re.compile(r'```\\s*$')\n"
        "\n"
        "def observer_node(state):\n"
        "    return {'ok': _FENCE_OPEN.match(state['text']) is not None}\n"
        "\n"
        "def check_import_conformance(pattern):\n"
        "    compiled = re.compile(pattern)\n"
        "    return compiled.match('x')\n",
        encoding="utf-8",
    )
    candidates = discover.scan_ast(tmp_path)
    specs = {c.spec.split(":")[-1] for c in candidates}
    assert "_FENCE_OPEN" not in specs
    assert "_FENCE_CLOSE" not in specs
    assert "observer_node" not in specs
    assert "check_import_conformance" not in specs
    assert candidates == []


def test_real_stategraph_compile_still_detected_alongside_unrelated_re_compile(tmp_path):
    src = tmp_path / "builder.py"
    src.write_text(
        "import re\n"
        "from langgraph.graph import StateGraph\n"
        "\n"
        "_PATTERN = re.compile(r'foo')\n"
        "\n"
        "def _compile_graph(checkpointer):\n"
        "    builder = StateGraph(dict)\n"
        "    builder.add_node('a', lambda s: s)\n"
        "    return builder.compile(checkpointer=checkpointer)\n",
        encoding="utf-8",
    )
    candidates = discover.scan_ast(tmp_path)
    specs = {c.spec.split(":")[-1] for c in candidates}
    assert "_compile_graph" in specs
    assert "_PATTERN" not in specs
