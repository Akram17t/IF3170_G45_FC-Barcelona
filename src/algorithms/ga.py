"""GA sesuai salindia: roulette wheel, crossover satu titik, mutasi; plus repair bottom-up dan elitism."""

from math import fsum
from time import perf_counter

from .common import configuration, initial_state
from ..initialize import random_feasible_state
from ..model import Placement, RunResult, State
from ..neighbors import sample_feasible_neighbor
from ..validation import objective, require_valid, validate


def roulette(population, scores, rng):
    """Peluang terpilih sebanding fitness; semua fitness nol -> pilih acak seragam."""
    total = fsum(scores)
    if total <= 0:
        return rng.choice(population)
    point, cumulative = rng.random() * total, 0.0
    for state, score in zip(population, scores):
        cumulative += score
        if point < cumulative:
            return state
    return population[-1]


def crossover(problem, first, second, rng):
    """Satu titik potong pada urutan ID paket; menghasilkan dua anak."""
    ids = [p.id for p in problem.packages]
    cut = rng.randrange(1, len(ids)) if len(ids) > 1 else 0
    return tuple(State({package_id: (a if index < cut else b).placements[package_id]
                        for index, package_id in enumerate(ids)})
                 for a, b in ((first, second), (second, first)))


def repair(problem, child, rng):
    """Satu pass bottom-up: pertahankan gen feasible, keluarkan gen lainnya."""
    placements = {p.id: Placement(None, child.placements[p.id].orientation) for p in problem.packages}
    state = State(placements)
    ids = [p.id for p in problem.packages if child.placements[p.id].position is not None]
    rng.shuffle(ids)
    ids.sort(key=lambda package_id: child.placements[package_id].position[2])
    dropped = 0
    for package_id in ids:
        candidate_map = dict(state.placements)
        candidate_map[package_id] = child.placements[package_id]
        candidate = State(candidate_map)
        if validate(problem, candidate).valid:
            state = candidate
        else:
            dropped += 1
    require_valid(problem, state)
    return state, dropped
