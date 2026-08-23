from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Criterion, Dimension, Severity


def load_rubric(path: str | Path) -> list[Criterion]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    criteria = []
    seen: set[str] = set()
    for item in data["criteria"]:
        criterion = Criterion(
            id=item["id"],
            dimension=Dimension(item["dimension"]),
            description=item["description"],
            severity=Severity(item["severity"]),
            deterministic_metric=item.get("deterministic_metric"),
        )
        if criterion.id in seen:
            raise ValueError(f"Duplicate criterion id: {criterion.id}")
        seen.add(criterion.id)
        criteria.append(criterion)
    return criteria


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def severity_frequency_weights(
    issue_history: list[dict[str, Any]],
    severity_multipliers: dict[str, float] | None = None,
    fallback_frequencies: dict[str, float] | None = None,
) -> dict[Dimension, float]:
    """Derive normalized dimension weights from historical issue frequency.

    Each observation contributes its count multiplied by its severity. This makes
    weights empirical while retaining the requested severity sensitivity.
    """
    multipliers = severity_multipliers or {
        "low": 1.0,
        "medium": 2.0,
        "high": 4.0,
        "critical": 8.0,
    }
    totals = {dimension: 0.0 for dimension in Dimension}
    for observation in issue_history:
        dimension = Dimension(observation["dimension"])
        count = max(0.0, float(observation.get("count", 1)))
        severity = str(observation.get("severity", "medium"))
        if severity not in multipliers:
            raise ValueError(f"Unknown severity in weight history: {severity}")
        totals[dimension] += count * multipliers[severity]

    if not any(totals.values()):
        frequencies = fallback_frequencies or {d.value: 1.0 for d in Dimension}
        totals = {d: max(0.0, float(frequencies.get(d.value, 0.0))) for d in Dimension}

    denominator = sum(totals.values())
    if denominator <= 0:
        raise ValueError("At least one dimension must have a positive frequency weight")
    return {dimension: value / denominator for dimension, value in totals.items()}


def renormalize_weights(
    weights: dict[Dimension, float], available: set[Dimension]
) -> dict[Dimension, float]:
    selected = {d: weight for d, weight in weights.items() if d in available}
    denominator = sum(selected.values())
    if denominator <= 0:
        raise ValueError("No weighted dimension produced a score")
    return {dimension: value / denominator for dimension, value in selected.items()}

