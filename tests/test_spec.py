from render_langgraph.spec import split_spec


def test_relative_path_with_attr():
    assert split_spec("graph.py:graph") == ("graph.py", "graph")


def test_relative_path_without_attr():
    assert split_spec("graph.py") == ("graph.py", "")


def test_windows_drive_path_with_attr():
    assert split_spec("C:/Users/x/graph.py:graph") == ("C:/Users/x/graph.py", "graph")
    assert split_spec("C:\\Users\\x\\graph.py:graph") == ("C:\\Users\\x\\graph.py", "graph")


def test_windows_drive_path_without_attr():
    assert split_spec("C:/Users/x/graph.py") == ("C:/Users/x/graph.py", "")
    assert split_spec("C:\\Users\\x\\graph.py") == ("C:\\Users\\x\\graph.py", "")


def test_nested_path_with_attr():
    assert split_spec("src/agent/graph.py:build_graph") == ("src/agent/graph.py", "build_graph")
