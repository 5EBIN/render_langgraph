"""The target module prints arbitrary non-JSON garbage at import time (e.g.
a logging setup banner, a deprecation warning some library prints itself,
etc). The extractor must still produce a clean parse -- it writes its
result to a temp file, never json.loads(stdout)."""
print("=== some banner a real project's imports might print ===")
print("not json at all: {not: valid, [")

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    text: str


def step(state: State) -> State:
    print("more stdout noise during node definition")
    return {"text": state["text"]}


builder = StateGraph(State)
builder.add_node("step", step)
builder.add_edge(START, "step")
builder.add_edge("step", END)

graph = builder.compile()
