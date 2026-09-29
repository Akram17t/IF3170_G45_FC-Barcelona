"""Validator bersama berdasarkan spesifikasi IF3170 v1.4, halaman 8-9."""

from itertools import combinations
from math import fsum, isclose

from .model import Number, Problem, State, ValidationResult, oriented_size


def validate(problem: Problem, state: State) -> ValidationResult:
    reasons = []
    expected = set(problem.package_map())
    actual = set(state.placements)
    if expected != actual:
        return ValidationResult(False, (
            f"ID tidak lengkap: kurang={sorted(expected - actual)}, asing={sorted(actual - expected)}",
        ))
    boxes = {}
    weight = []
    for package in problem.packages:
        placement = state.placements[package.id]
        if placement.position is None:
            continue
        lower = placement.position
        upper = tuple(a + b for a, b in zip(lower, oriented_size(package, placement.orientation)))
        if any(a < 0 or b > limit for a, b, limit in zip(lower, upper, problem.truck.dimensions)):
            reasons.append(f"{package.id}: menembus dinding")
        boxes[package.id] = (lower, upper, package.is_fragile)
        weight.append(package.weight)
    total = fsum(weight)
    capacity = problem.truck.max_capacity
    if total > capacity and not isclose(total, capacity, rel_tol=1e-12, abs_tol=1e-12):
        reasons.append(f"Berat {total:g} melebihi kapasitas {capacity:g}")
    for (a_id, (a, aa, _)), (b_id, (b, bb, _)) in combinations(boxes.items(), 2):
        if all(max(a[i], b[i]) < min(aa[i], bb[i]) for i in range(3)):
            reasons.append(f"{a_id} dan {b_id}: overlap")
    for package_id, (lower, upper, _) in boxes.items():
        if lower[2] == 0:
            continue
        supporters = []
        for other_id, (other, top, fragile) in boxes.items():
            if other_id == package_id or top[2] != lower[2]:
                continue
            if all(max(lower[i], other[i]) < min(upper[i], top[i]) for i in (0, 1)):
                supporters.append(other_id)
                if fragile:
                    reasons.append(f"{package_id}: ditopang paket fragile {other_id}")
        if not supporters:
            reasons.append(f"{package_id}: melayang")
    return ValidationResult(not reasons, tuple(reasons))


def require_valid(problem: Problem, state: State) -> None:
    result = validate(problem, state)
    if not result.valid:
        raise ValueError("State tidak feasible: " + "; ".join(result.reasons))


def total_weight(problem: Problem, state: State) -> Number:
    return fsum(p.weight for p in problem.packages if state.placements[p.id].position is not None)


def objective(problem: Problem, state: State) -> Number:
    require_valid(problem, state)
    return fsum(p.value for p in problem.packages if state.placements[p.id].position is not None)
