"""Break a mined gplearn program down into its evaluated sub-expressions for
one specific row of data -- "the why" behind a single stock's score.

Mirrors gplearn's own stack-based `_Program.execute` exactly (same
_Function objects, same protected operators) instead of re-parsing the
formula string, so there's no risk of the breakdown's numbers silently
diverging from what the model actually computed.
"""
import numpy as np
from gplearn.functions import _Function


def breakdown_program(program, X_row: np.ndarray, feature_names: list[str]) -> dict:
    """X_row: shape (1, n_features) -- a single stock's feature vector.
    Returns a nested {label, value, children} tree."""

    def _terminal_info(t):
        if isinstance(t, float):
            return {"label": f"{t:.4g}", "value": t, "children": []}
        if isinstance(t, int):
            return {"label": feature_names[t], "value": float(X_row[0, t]), "children": []}
        return t  # already a computed node dict

    node0 = program.program[0]
    if isinstance(node0, float):
        return _terminal_info(node0)
    if isinstance(node0, int):
        return _terminal_info(node0)

    apply_stack = []
    for node in program.program:
        if isinstance(node, _Function):
            apply_stack.append([node])
        else:
            apply_stack[-1].append(node)

        while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
            function = apply_stack[-1][0]
            terminal_infos = [_terminal_info(t) for t in apply_stack[-1][1:]]
            values = [np.array([info["value"]]) for info in terminal_infos]
            result_val = float(function(*values)[0])
            label = f"{function.name}(" + ", ".join(info["label"] for info in terminal_infos) + ")"
            node_info = {"label": label, "value": result_val, "children": terminal_infos}
            if len(apply_stack) != 1:
                apply_stack.pop()
                apply_stack[-1].append(node_info)
            else:
                return node_info

    raise RuntimeError("Malformed program: never resolved to a single result")


def render_tree_lines(node: dict, prefix: str = "", is_last: bool = True, is_root: bool = True) -> list[str]:
    """Render a breakdown_program() tree as ASCII-art lines, e.g.:
    sub(sum_volume_10, delta_close_5) = -0.3997
    ├─ sum_volume_10 = -0.6301
    └─ delta_close_5 = -0.2304
    """
    connector = "" if is_root else ("└─ " if is_last else "├─ ")
    lines = [f"{prefix}{connector}{node['label']} = {node['value']:.4g}"]
    children = node.get("children", [])
    new_prefix = prefix if is_root else prefix + ("   " if is_last else "│  ")
    for i, child in enumerate(children):
        lines.extend(render_tree_lines(child, new_prefix, i == len(children) - 1, is_root=False))
    return lines
