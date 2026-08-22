"""Reproduces the real-world layout: a `core` package that's a sibling of
`app/` (imported unqualified, `from core...`) AND a `core` subpackage of
`app` itself (imported qualified, `from app.core...`). Both must resolve
whether langgraphv is run from inside app/ or from the project root --
the extractor has to put the actual project root (found by walking up for
pyproject.toml/setup.py/setup.cfg/.git), not just cwd, on sys.path, and
import this file BY DOTTED NAME (app.service) rather than by raw file path
so `app` is a real parent package."""
from typing import TypedDict

from app.core.extra import EXTRA
from core.config import GREETING
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    text: str


def step(state: State) -> State:
    return {"text": GREETING + EXTRA + state["text"]}


builder = StateGraph(State)
builder.add_node("step", step)
builder.add_edge(START, "step")
builder.add_edge("step", END)

graph = builder.compile()
