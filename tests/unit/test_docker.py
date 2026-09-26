import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOP_LEVEL = {"configs", "ingest", "features", "index", "governance", "api", "change", "worker", "query"}


def _imports(pkg: str) -> set[str]:
    found = set()
    for f in (REPO / pkg).rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.add(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                found.update(a.name.split(".")[0] for a in node.names)
    return found & TOP_LEVEL


def test_api_image_copies_every_package_the_services_import():
    """The api image COPYs packages one by one; a missing one only fails at request time (M4: query/)."""
    copied = set(
        re.findall(r"^COPY (\w+) \1$", (REPO / "docker" / "Dockerfile.api").read_text(), re.MULTILINE)
    )
    needed, todo = set(), {"api", "worker"}  # both services run this image
    while todo:  # transitive closure over our own packages
        pkg = todo.pop()
        needed.add(pkg)
        todo |= _imports(pkg) - needed
    assert needed <= copied, f"missing COPY in docker/Dockerfile.api: {sorted(needed - copied)}"
