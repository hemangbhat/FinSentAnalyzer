"""
Static smoke test: every symbol a Streamlit page imports from a project module
must actually exist in that module.

This catches broken imports (e.g. `from preprocess import LABEL_MAP_INV`, which
never existed there) without launching Streamlit. Optional heavy dependencies
(torch, shap, transformers, streamlit) that aren't installed are tolerated by
skipping the affected module, so the test stays green in minimal CI envs while
still validating the import surface wherever the deps are present.
"""

import ast
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
APP = ROOT / "app"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(APP))

# Project modules whose imports we validate (src modules + app-level helpers).
PROJECT_MODULES = {p.stem for p in SRC.glob("*.py")} | {"shared", "ui"}

PAGE_FILES = sorted((APP / "pages").glob("*.py")) + [APP / "app.py", APP / "shared.py"]


def _project_imports(path: Path):
    """Yield (module, [imported_names]) for imports from project modules."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in PROJECT_MODULES:
            names = [alias.name for alias in node.names if alias.name != "*"]
            if names:
                yield node.module, names


@pytest.mark.parametrize("page", PAGE_FILES, ids=lambda p: p.name)
def test_page_imports_resolve(page: Path):
    missing: list[str] = []
    for module, names in _project_imports(page):
        try:
            mod = importlib.import_module(module)
        except ModuleNotFoundError:
            # Optional heavy dependency (torch/shap/transformers/streamlit) not
            # installed in this environment — can't validate, so skip.
            continue
        for name in names:
            if not hasattr(mod, name):
                missing.append(f"{module}.{name}")
    assert not missing, f"{page.name} imports symbols that don't exist: {missing}"
