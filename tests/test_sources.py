"""Every source file in the repository has to at least compile.

Not a style check. `tools/evolve/__main__.py` sat in the repository for several
commits with a literal newline inside a string literal -- the whole evolution
entry point raised SyntaxError before argparse ever ran -- and nothing noticed,
because no test imports a `__main__`, and the modules that are imported were
fine. A file nothing imports is exactly the file that breaks quietly.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TREES = ["stonkfly", "tools", "tests"]


def sources():
    for tree in TREES:
        for path in sorted((ROOT / tree).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path


@pytest.mark.parametrize("path", list(sources()),
                         ids=lambda p: str(p.relative_to(ROOT)).replace("\\", "/"))
def test_compiles(path):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
