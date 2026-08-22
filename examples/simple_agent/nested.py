"""Happy path: a compiled subgraph used as a node in a parent graph."""
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class SubState(TypedDict):
    text: str


def sub_step(state: SubState) -> SubState:
    return {"text": state["text"] + "-sub"}


sub_builder = StateGraph(SubState)
sub_builder.add_node("sub_step", sub_step)
sub_builder.add_edge(START, "sub_step")
sub_builder.add_edge("sub_step", END)
researcher = sub_builder.compile()


class State(TypedDict):
    text: str


def intro(state: State) -> State:
    return {"text": "start-" + state["text"]}


builder = StateGraph(State)
builder.add_node("intro", intro)
builder.add_node("researcher", researcher)
builder.add_edge(START, "intro")
builder.add_edge("intro", "researcher")
builder.add_edge("researcher", END)

graph = builder.compile()
