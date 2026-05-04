from __future__ import annotations

import re

import numpy as np
from sklearn.tree import DecisionTreeClassifier


def _format_threshold(value: float) -> str:
    formatted = f"{value:.6g}"
    if "e" not in formatted and "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    if formatted in {"-0", "-0."}:
        formatted = "0"
    return formatted


def _sanitize_identifier(name: str, fallback_index: int) -> str:
    candidate = re.sub(r"[^0-9A-Za-z_]", "_", name).upper().strip("_")
    if not candidate:
        return f"CLASS_{fallback_index}"
    if candidate[0].isdigit():
        candidate = f"CLASS_{candidate}"
    return candidate


def _build_class_identifiers(class_names: list[str], model: DecisionTreeClassifier) -> list[str]:
    def _dedupe(identifiers: list[str]) -> list[str]:
        seen: dict[str, int] = {}
        unique_identifiers: list[str] = []
        for identifier in identifiers:
            count = seen.get(identifier, 0)
            unique_identifier = identifier if count == 0 else f"{identifier}_{count + 1}"
            seen[identifier] = count + 1
            unique_identifiers.append(unique_identifier)
        return unique_identifiers

    if class_names:
        return _dedupe([_sanitize_identifier(name, index) for index, name in enumerate(class_names)])
    class_count = int(getattr(model, "n_classes_", 0) or 0)
    return [f"CLASS_{index}" for index in range(class_count)]


def _class_label(node_value: np.ndarray, class_identifiers: list[str]) -> str:
    class_index = int(np.argmax(node_value[0]))
    if class_identifiers and 0 <= class_index < len(class_identifiers):
        return class_identifiers[class_index]
    return f"CLASS_{class_index}"


def _export_node(node_id: int, tree, feature_prefix: str, class_identifiers: list[str], indent: int = 1) -> str:
    padding = "    " * indent
    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]
    if left_child == right_child:
        return f"{padding}return {_class_label(tree.value[node_id], class_identifiers)};\n"

    feature_index = tree.feature[node_id]
    threshold = _format_threshold(float(tree.threshold[node_id]))
    output = f"{padding}if ({feature_prefix}[{feature_index}] <= {threshold}) {{\n"
    output += _export_node(left_child, tree, feature_prefix, class_identifiers, indent + 1)
    output += f"{padding}}} else {{\n"
    output += _export_node(right_child, tree, feature_prefix, class_identifiers, indent + 1)
    output += f"{padding}}}\n"
    return output


def export_tree_to_c(
    model: DecisionTreeClassifier,
    class_names: list[str] | None = None,
    function_name: str = "predict_command",
    feature_prefix: str = "f",
) -> str:
    if class_names is None:
        class_names = []

    tree = model.tree_
    class_identifiers = _build_class_identifiers(class_names, model)
    lines = ["#include <stdint.h>", ""]
    if class_identifiers:
        lines.append("typedef enum {")
        for index, identifier in enumerate(class_identifiers):
            suffix = "," if index < len(class_identifiers) - 1 else ""
            lines.append(f"    {identifier}{suffix}")
        lines.append(f"}} {function_name}_class_t;")
        lines.append("")
    lines.append(f"int {function_name}(const float *{feature_prefix}) {{")
    lines.append(_export_node(0, tree, feature_prefix, class_identifiers, indent=1).rstrip())
    lines.append("}")
    if class_names:
        lines.append("")
        lines.append("/* Class order:")
        for index, name in enumerate(class_names):
            lines.append(f"   {index}: {name}")
        lines.append("*/")
    return "\n".join(lines)
