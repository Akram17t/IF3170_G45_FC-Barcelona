# acak yang feasible (jumlah percobaan masih terbatas)
from random import Random

from .model import Orientation, Placement, Problem, State, oriented_size
from .validation import validate


def candidate_position(problem, state, package, orientation, rng):
    size = oriented_size(package, orientation)
    limits = tuple(a - b for a, b in zip(problem.truck.dimensions, size))
    if min(limits) < 0:
        return None
    # tiap posisi feasible di atas lantai tepat setinggi atap penyangganya
    levels = {0}
    for other in problem.packages:
        place = state.placements[other.id]
        if place.position is not None and not other.is_fragile:
            top = place.position[2] + oriented_size(other, place.orientation)[2]
            if top <= limits[2]:
                levels.add(top)
    return rng.randrange(limits[0] + 1), rng.randrange(limits[1] + 1), rng.choice(sorted(levels))


def random_feasible_state(problem: Problem, rng: Random, max_attempts: int = 100) -> State:
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts <= 0:
        raise ValueError("max_attempts harus integer positif")
    state = State.empty(problem)
    packages = list(problem.packages)
    rng.shuffle(packages)
    for package in packages:
        for _ in range(max_attempts):
            orientation = rng.choice(list(Orientation))
            position = candidate_position(problem, state, package, orientation, rng)
            if position is None:
                continue
            placements = dict(state.placements)
            placements[package.id] = Placement(position, orientation)
            candidate = State(placements)
            if validate(problem, candidate).valid:
                state = candidate
                break
    return state
