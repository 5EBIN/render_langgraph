"""Failure: config_error -- top-level Settings() validation needs an env var
that isn't set. Also exercises the extractor's auto .env retry: if a .env
next to this file defines DATABASE_URL, import succeeds without the user
doing anything."""
from pydantic_settings import BaseSettings

from langgraph.graph import StateGraph, START, END


class Settings(BaseSettings):
    database_url: str


settings = Settings()


class State(dict):
    pass


def step(state):
    return {"text": state.get("text", "") + settings.database_url}


builder = StateGraph(dict)
builder.add_node("step", step)
builder.add_edge(START, "step")
builder.add_edge("step", END)

graph = builder.compile()
