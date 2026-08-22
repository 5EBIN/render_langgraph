"""Covers the "wrong interpreter" and "ModuleNotFoundError for sibling
packages" bugs directly: a project laid out as app/service.py doing both
`from app.core import x` (subpackage of app) and `from core import x`
(top-level sibling of app), run from OUTSIDE app/ (cwd = project root)."""
from render_langgraph.extract import extract, prepare_target


def test_dual_style_imports_resolve_after_root_injection(nested_project_python, nested_project):
    result = extract(nested_project_python, "app/service.py:graph", nested_project)
    assert result.ok, result.data
    node_ids = {n["id"] for n in result.data["nodes"]}
    assert "step" in node_ids


def test_module_is_imported_by_dotted_name_not_file_path(nested_project):
    target = prepare_target("app/service.py:graph", nested_project)
    assert target.module_name == "app.service"
    assert target.project_root == nested_project.resolve()


def test_project_root_found_via_pyproject_toml_not_cwd(nested_project):
    # cwd is the project root itself here, but the important property is
    # that project_root doesn't just default to the target file's own
    # directory (app/) -- it must be found by walking up to the anchor.
    target = prepare_target("app/service.py:graph", nested_project)
    assert target.project_root == nested_project.resolve()
    assert target.project_root != (nested_project / "app").resolve()
