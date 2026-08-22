"""Fixtures for the branch-classification feature: one compiled graph per
router pattern we need to distinguish. Each router's classification should
be derivable purely from its source (inspect.getsource + ast) -- none of
these actually call a real LLM; the classifier only looks at call *shape*
(e.g. `something.invoke(...)`), so a stand-in object with an `invoke`
method is enough to exercise the "dynamic" signal honestly.

Node ids are namespaced per pattern (route_a_1, route_a_2, ...) rather than
reused as bare "a"/"b" -- static_parse.py scans this whole file as one flat
AST with no notion of "which StateGraph a call belongs to", so reusing the
same node id across independent graphs in one file would make the last
add_conditional_edges("a", ...) call silently overwrite the classification
recorded for all the earlier ones. Runtime extraction doesn't have this
problem (each graph_* compiles independently), but the fixture needs to be
static-analysis-safe too since static_parse.py is tested against this file.
"""
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    x: int
    history: list[str]


def _step(name: str):
    def fn(state: State) -> State:
        return {"history": state["history"] + [name]}

    return fn


# --- 1. deterministic: Literal return, branches only on state, no model call ---
det_a = _step("det_a")
det_b = _step("det_b")


def route_deterministic(state: State) -> Literal["det_a", "det_b"]:
    if state["x"] > 0:
        return "det_a"
    return "det_b"


builder_deterministic = StateGraph(State)
builder_deterministic.add_node("det_a", det_a)
builder_deterministic.add_node("det_b", det_b)
builder_deterministic.add_edge(START, "det_a")
builder_deterministic.add_conditional_edges("det_a", route_deterministic)
builder_deterministic.add_edge("det_b", END)
graph_deterministic = builder_deterministic.compile()


# --- 2. dynamic: router calls llm.invoke(...) and returns its output ---
class _FakeLLM:
    def invoke(self, prompt):
        return prompt


llm = _FakeLLM()
dyn_a = _step("dyn_a")
dyn_b = _step("dyn_b")


def route_dynamic_llm(state: State) -> Literal["dyn_a", "dyn_b"]:
    response = llm.invoke(state["history"])
    return "dyn_a" if response else "dyn_b"


builder_dynamic_llm = StateGraph(State)
builder_dynamic_llm.add_node("dyn_a", dyn_a)
builder_dynamic_llm.add_node("dyn_b", dyn_b)
builder_dynamic_llm.add_edge(START, "dyn_a")
builder_dynamic_llm.add_conditional_edges("dyn_a", route_dynamic_llm)
builder_dynamic_llm.add_edge("dyn_b", END)
graph_dynamic_llm = builder_dynamic_llm.compile()


# --- 3. dynamic: no Literal/path_map hints at all, open-ended model output ---
chain = _FakeLLM()
nohint_a = _step("nohint_a")
nohint_b = _step("nohint_b")


def route_dynamic_no_hints(state: State):
    return chain.invoke(state["history"])


builder_dynamic_no_hints = StateGraph(State)
builder_dynamic_no_hints.add_node("nohint_a", nohint_a)
builder_dynamic_no_hints.add_node("nohint_b", nohint_b)
builder_dynamic_no_hints.add_edge(START, "nohint_a")
builder_dynamic_no_hints.add_conditional_edges("nohint_a", route_dynamic_no_hints)
builder_dynamic_no_hints.add_edge("nohint_b", END)
graph_dynamic_no_hints = builder_dynamic_no_hints.compile()


# --- 4. unknown: router built dynamically (exec'd), source not resolvable ---
_dynamic_src = (
    "def _built_at_runtime(state):\n"
    "    return 'dynbuilt_a' if state['x'] > 0 else 'dynbuilt_b'\n"
)
_dynamic_ns: dict = {}
exec(compile(_dynamic_src, "<dynamically-generated>", "exec"), _dynamic_ns)
route_unknown_dynamic = _dynamic_ns["_built_at_runtime"]

dynbuilt_a = _step("dynbuilt_a")
dynbuilt_b = _step("dynbuilt_b")

builder_unknown_dynamic = StateGraph(State)
builder_unknown_dynamic.add_node("dynbuilt_a", dynbuilt_a)
builder_unknown_dynamic.add_node("dynbuilt_b", dynbuilt_b)
builder_unknown_dynamic.add_edge(START, "dynbuilt_a")
builder_unknown_dynamic.add_conditional_edges("dynbuilt_a", route_unknown_dynamic)
builder_unknown_dynamic.add_edge("dynbuilt_b", END)
graph_unknown_dynamic = builder_unknown_dynamic.compile()


# --- 5. unknown: lambda router, no inspectable body ---
lam_a = _step("lam_a")
lam_b = _step("lam_b")

builder_unknown_lambda = StateGraph(State)
builder_unknown_lambda.add_node("lam_a", lam_a)
builder_unknown_lambda.add_node("lam_b", lam_b)
builder_unknown_lambda.add_edge(START, "lam_a")
builder_unknown_lambda.add_conditional_edges("lam_a", lambda state: "lam_a" if state["x"] > 0 else "lam_b")
builder_unknown_lambda.add_edge("lam_b", END)
graph_unknown_lambda = builder_unknown_lambda.compile()
