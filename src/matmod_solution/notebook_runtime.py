from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable


def load_notebook_namespace(
    notebook_path: str | Path,
    code_cell_indexes: Iterable[int],
) -> SimpleNamespace:
    """Execute selected notebook code cells and expose the resulting namespace."""

    path = Path(notebook_path)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__notebook_runtime__"}

    for idx in code_cell_indexes:
        source = "".join(notebook["cells"][idx].get("source", []))
        exec(compile(source, f"{path.name}:cell_{idx}", "exec"), namespace)

    return SimpleNamespace(**namespace)

