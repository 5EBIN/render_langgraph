"""Synthetic stand-in for the real "your_store" orchestration graph used to
drive the P1-P5 layout work: a flat StateGraph (no real compiled subgraphs,
so this is layout Case B), ~20 nodes, 5 router nodes each with >=2
conditional out-edges, and multiple cycles (capabilities->capability,
enough->plan) so entry/end anchoring and back-edge handling are exercised."""
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    step: int
    history: list[str]


def _make(name: str):
    def fn(state: State) -> State:
        return {"history": state["history"] + [name], "step": state["step"] + 1}

    return fn


# --- router nodes: do a little work, then branch ---
entry = _make("entry")
capability = _make("capability")
capabilities = _make("capabilities")
plan = _make("plan")
enough = _make("enough")

# --- worker nodes ---
cap_worker_a = _make("cap_worker_a")
cap_worker_b = _make("cap_worker_b")
list_caps = _make("list_caps")
describe_cap = _make("describe_cap")
scan_plan = _make("scan_plan")
run_verified_query = _make("run_verified_query")
gate = _make("gate")
execute = _make("execute")
finalize = _make("finalize")
validate = _make("validate")
summarize = _make("summarize")
notify = _make("notify")
cleanup = _make("cleanup")
log_result = _make("log_result")
persist = _make("persist")


def route_entry(state: State) -> Literal["capability", "capabilities", "plan"]:
    n = state["step"] % 3
    return ["capability", "capabilities", "plan"][n]


def route_capability(state: State) -> Literal["cap_worker_a", "cap_worker_b", "capabilities"]:
    n = state["step"] % 3
    return ["cap_worker_a", "cap_worker_b", "capabilities"][n]


def route_capabilities(state: State) -> Literal["list_caps", "describe_cap", "capability"]:
    n = state["step"] % 3
    return ["list_caps", "describe_cap", "capability"][n]


def route_plan(state: State) -> Literal["scan_plan", "run_verified_query", "gate"]:
    n = state["step"] % 3
    return ["scan_plan", "run_verified_query", "gate"][n]


def route_enough(state: State) -> Literal["plan", "finalize", "__end__"]:
    if state["step"] > 12:
        return "finalize" if state["step"] % 2 == 0 else "__end__"
    return "plan"


builder = StateGraph(State)
for name, fn in [
    ("entry", entry),
    ("capability", capability),
    ("capabilities", capabilities),
    ("plan", plan),
    ("enough", enough),
    ("cap_worker_a", cap_worker_a),
    ("cap_worker_b", cap_worker_b),
    ("list_caps", list_caps),
    ("describe_cap", describe_cap),
    ("scan_plan", scan_plan),
    ("run_verified_query", run_verified_query),
    ("gate", gate),
    ("execute", execute),
    ("finalize", finalize),
    ("validate", validate),
    ("summarize", summarize),
    ("notify", notify),
    ("cleanup", cleanup),
    ("log_result", log_result),
    ("persist", persist),
]:
    builder.add_node(name, fn)

builder.add_edge(START, "entry")
builder.add_conditional_edges("entry", route_entry)
builder.add_conditional_edges("capability", route_capability)
builder.add_conditional_edges("capabilities", route_capabilities)  # -> capability is a back edge (cycle)
builder.add_conditional_edges("plan", route_plan)
builder.add_conditional_edges("enough", route_enough)  # -> plan is a back edge (cycle)

builder.add_edge("cap_worker_a", "validate")
builder.add_edge("cap_worker_b", "validate")
builder.add_edge("validate", "enough")

builder.add_edge("list_caps", "summarize")
builder.add_edge("describe_cap", "summarize")
builder.add_edge("summarize", "enough")

builder.add_edge("scan_plan", "execute")
builder.add_edge("run_verified_query", "gate")
builder.add_edge("execute", "log_result")
builder.add_edge("gate", "log_result")  # log_result is reachable from two plan-side workers equally -> ambiguous owner, must not cluster
builder.add_edge("log_result", "enough")

builder.add_edge("finalize", "notify")
builder.add_edge("notify", "cleanup")
builder.add_edge("cleanup", "persist")
builder.add_edge("persist", END)

graph = builder.compile()
