"""Experimental v2 geometry checks. The existing default judge is unchanged.

Spatial requirements must be frozen from the description, never from candidates.
Topology matching only accepts a unique full directed-graph correspondence;
ambiguous or incomplete graphs fall back to the original evidence-based matcher.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import math

from .deterministic import (
    ComponentMatch, DeterministicJudge, _area, _contains_bounds,
    _labeled_shapes, _nearest_key, _normalize_label, _overlap_fraction,
)
from .models import Issue, Severity


class SensitivityJudge(DeterministicJudge):
    name = "deterministic-svg-v2"

    def match_components(self, candidate, expectations):
        # A title on a full-canvas background is not a drawn component. In a
        # blank candidate, lexical similarity must not turn that background
        # into a missing group or node.
        foreground = type(candidate)(candidate.width, candidate.height, tuple(
            e for e in candidate.elements if not (e.is_shape and e.bounds.x <= 0
            and e.bounds.y <= 0 and e.bounds.right >= candidate.width
            and e.bounds.bottom >= candidate.height)
        ))
        matched = super().match_components(foreground, expectations)
        edges = {(e["source"], e["target"]) for e in expectations.get("connections", [])}
        connected = {node for edge in edges for node in edge}
        if not connected or len(connected) > 20:
            return matched
        shapes = [e for e in candidate.elements if e.is_shape]
        # Full graph correspondence is only attempted on leaf shapes. Explicit
        # matched group labels and full-canvas backgrounds cannot be graph nodes.
        nodes = [s for s in shapes if not any(
            s is not other and _contains_bounds(s.bounds, other.bounds, 0.1)
            and _area(s.bounds) > _area(other.bounds) for other in shapes
        )]
        pool = [(s.id, s) for s in nodes]
        threshold = self.config.endpoint_tolerance * math.hypot(candidate.width, candidate.height)
        actual = []
        for connector in candidate.elements:
            if connector.is_connector and len(connector.points) >= 2:
                first = _nearest_key(connector.directed_points[0], pool, threshold)
                last = _nearest_key(connector.directed_points[-1], pool, threshold)
                if first is None or last is None:
                    return matched
                actual.append((first, last))
        actual_edges = set(actual)
        actual_nodes = {node for edge in actual_edges for node in edge}
        if len(actual) != len(edges) or len(actual_nodes) != len(connected):
            return matched
        components = {c["id"]: c for c in expectations["components"]}
        frequencies = Counter(_normalize_label(c["label"]) for c in components.values())
        labels = {p.shape.id: p.text for p in _labeled_shapes(candidate)}
        # Only unique exact visible labels anchor graph roles. Duplicate labels
        # (e.g. two Add & Norm nodes) must be distinguished by their connections.
        anchors = {cid: m.shape.id for cid, m in matched.items()
                   if cid in connected and m.text and frequencies[_normalize_label(components[cid]["label"])] == 1
                   and _normalize_label(m.text.text) == _normalize_label(components[cid]["label"])}
        degree = lambda node, graph: (sum(b == node for a, b in graph), sum(a == node for a, b in graph))
        candidates = {cid: [node for node in actual_nodes
                            if degree(cid, edges) == degree(node, actual_edges)
                            and (cid not in anchors or node == anchors[cid])]
                      for cid in connected}
        solutions = []
        visits = 0

        def search(mapping):
            nonlocal visits
            visits += 1
            if visits > 10000 or len(solutions) > 1:
                return
            if len(mapping) == len(connected):
                solutions.append(dict(mapping))
                return
            cid = min((c for c in connected if c not in mapping), key=lambda c: (len(candidates[c]), c))
            for node in sorted(candidates[cid]):
                if node in mapping.values():
                    continue
                if ((cid, cid) in edges) != ((node, node) in actual_edges):
                    continue
                if any(((cid, other) in edges) != ((node, mapped) in actual_edges)
                       or ((other, cid) in edges) != ((mapped, node) in actual_edges)
                       for other, mapped in mapping.items()):
                    continue
                mapping[cid] = node
                search(mapping)
                del mapping[cid]

        search({})
        if len(solutions) != 1 or visits > 10000:
            return matched
        shapes_by_id = {s.id: s for s in nodes}
        for cid, node in solutions[0].items():
            text = labels.get(node)
            correct = text and _normalize_label(text.text) == _normalize_label(components[cid]["label"])
            matched[cid] = ComponentMatch(shapes_by_id[node], text, "label" if correct else "role",
                                          1.0 if correct else 0.0,
                                          "Unique correspondence of the complete visible directed graph; label correctness checked separately")
        return matched

    def evaluate(self, candidate, criteria, expectations):
        matched = self.match_components(candidate, expectations)
        metrics = {m.criterion_id: m for m in super().evaluate(candidate, criteria, expectations)}
        spatial = expectations.get("spatial_requirements", {})
        handlers = {
            "description_order": lambda: self.order(spatial, matched),
            "spatial_relations": lambda: self.relations(spatial, matched),
            "group_containment": lambda: self.containment(spatial, matched),
            "unintended_overlap": lambda: self.overlap(candidate, expectations, matched),
            "connector_attachment": lambda: self.attachments(candidate, expectations, matched),
        }
        for criterion in criteria:
            if criterion.deterministic_metric in handlers:
                result = handlers[criterion.deterministic_metric]()
                result.criterion_id, result.dimension = criterion.id, criterion.dimension
                metrics[criterion.id] = result
        return list(metrics.values())

    def order(self, spatial, matched):
        issues, total = [], 0
        for sequence in spatial.get("sequences", []):
            axis = sequence["axis"]
            if axis not in {"top_to_bottom", "bottom_to_top", "left_to_right", "right_to_left"}:
                raise ValueError(f"Unsupported spatial axis: {axis}")
            coordinate = 1 if axis in {"top_to_bottom", "bottom_to_top"} else 0
            sign = -1 if axis in {"bottom_to_top", "right_to_left"} else 1
            for first, last in combinations(sequence["components"], 2):
                if first not in matched or last not in matched:
                    continue  # Absence belongs to Presence; not a successful order check.
                total += 1
                delta = sign * (matched[last].shape.bounds.center[coordinate] - matched[first].shape.bounds.center[coordinate])
                if delta <= 5:
                    issues.append(Issue("Required sequence pair is reversed or lacks spatial separation", Severity.MEDIUM, [first, last]))
        return self._score(total, issues, "Pairwise order agreement within explicitly stated sequences. No default direction is inferred from arrows; missing spatial requirements are N/A.")

    def relations(self, spatial, matched):
        issues, total = [], 0
        for relation in spatial.get("relations", []):
            first, last = relation["first"], relation["second"]
            if first not in matched or last not in matched:
                continue
            a, b = matched[first].shape.bounds, matched[last].shape.bounds
            tests = {
                "above": a.center[1] + 5 < b.center[1],
                "left_of": a.center[0] + 5 < b.center[0],
                "same_row": abs(a.center[1] - b.center[1]) <= min(a.height, b.height) / 2,
                "same_column": abs(a.center[0] - b.center[0]) <= min(a.width, b.width) / 2,
            }
            kind = relation["relation"]
            if kind not in tests:
                raise ValueError(f"Unsupported spatial relation: {kind}")
            total += 1
            if not tests[kind]:
                issues.append(Issue(f"Description requires {first} {kind} {last}", Severity.MEDIUM, [first, last]))
        return self._score(total, issues, "Only spatial relations explicitly frozen from the description are tested.")

    def containment(self, spatial, matched):
        issues, total = [], 0
        for group in spatial.get("groups", []):
            container = group["container"]
            for member in group["members"]:
                if container not in matched or member not in matched:
                    continue
                total += 1
                if not _contains_bounds(matched[container].shape.bounds, matched[member].shape.bounds, self.config.canvas_tolerance):
                    issues.append(Issue("Required member lies outside its intended group", Severity.HIGH, [container, member]))
        return self._score(total, issues)

    def overlap(self, candidate, expectations, matched):
        group_ids = {g["container"] for g in expectations.get("spatial_requirements", {}).get("groups", [])}
        group_ids.update(c["id"] for c in expectations.get("components", []) if c.get("kind") in {"group", "container"})
        containers = {matched[c].shape.id for c in group_ids if c in matched}
        leaves = [s for s in candidate.elements if s.is_shape and s.id not in containers
                  and not (s.bounds.x <= 0 and s.bounds.y <= 0 and s.bounds.right >= candidate.width and s.bounds.bottom >= candidate.height)]
        collided, evidence = set(), {}
        for a, b in combinations(leaves, 2):
            if _overlap_fraction(a.bounds, b.bounds) > self.config.overlap_area_threshold:
                collided.update([a.id, b.id])
                evidence.setdefault(a.id, b.id)
                evidence.setdefault(b.id, a.id)
        return self._score(len(leaves), [Issue("Component collides with another non-container component", Severity.HIGH, [key, evidence[key]]) for key in sorted(collided)],
                           "Each affected component fails once. Enclosing another node does not automatically make a shape an intentional container.")

    def attachments(self, candidate, expectations, matched):
        expected = {(e["source"], e["target"]) for e in expectations.get("connections", [])}
        actual = self._actual_connections(candidate, matched)
        issues, seen = [], set()
        for first, last, connector in actual:
            if (first, last) not in expected or (first, last) in seen:
                issues.append(Issue("Arrow has an incorrect, unattached, reversed, or duplicate source-to-target pair", Severity.CRITICAL, [connector.id]))
            seen.add((first, last))
        return self._score(len(actual), issues, "Checks semantic source/target pairs as well as physical attachment; zero visible arrows is N/A, not perfect.")
