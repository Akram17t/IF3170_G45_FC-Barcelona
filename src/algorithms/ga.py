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


def run(problem, config, rng, initial=None):
    config = configuration("ga", config)
    start = perf_counter()
    size = config["population_size"]
    population = [initial_state(problem, rng, initial)]
    population.extend(random_feasible_state(problem, rng) for _ in range(size - 1))
    scores = [objective(problem, state) for state in population]
    first = best = population[max(range(size), key=lambda i: scores[i])]
    first_score = best_score = max(scores)
    initial_population = [state.to_dict() for state in population]
    initial_scores = scores[:]
    maxima, means, sizes = [max(scores)], [fsum(scores) / size], [size]
    dropped = mutations = proposals = invalid = 0
    for _ in range(config["generations"]):
        ranked = sorted(range(size), key=lambda i: scores[i], reverse=True)
        following = [population[i] for i in ranked[:config["elitism"]]]
        while len(following) < size:
            first_parent = roulette(population, scores, rng)
            second_parent = roulette(population, scores, rng)
            for raw in crossover(problem, first_parent, second_parent, rng):
                if len(following) == size:
                    break
                child, lost = repair(problem, raw, rng)
                dropped += lost
                if rng.random() < config["mutation_rate"]:
                    neighbor = sample_feasible_neighbor(problem, child, rng, config["max_attempts"])
                    proposals += neighbor.attempts
                    invalid += neighbor.rejected
                    if neighbor.state is not None:
                        child = neighbor.state
                        mutations += 1
                following.append(child)
        population = following
        scores = [objective(problem, state) for state in population]
        winner = max(range(size), key=lambda i: scores[i])
        if scores[winner] > best_score:
            best, best_score = population[winner], scores[winner]
        maxima.append(max(scores))
        means.append(fsum(scores) / size)
        sizes.append(len(population))
    final_index = max(range(size), key=lambda i: scores[i])
    return RunResult(
        "ga", config["seed"], config, first, population[final_index], first_score, scores[final_index],
        tuple(maxima), perf_counter() - start, "generation_limit", best, best_score,
        {"generations": config["generations"], "population_size": size,
         "population_sizes": sizes, "max_history": maxima, "mean_history": means,
         "initial_population": initial_population, "initial_population_scores": initial_scores,
         "final_population": [state.to_dict() for state in population], "final_population_scores": scores,
         "repair_dropped_genes": dropped, "mutations": mutations, "proposals": proposals,
         "invalid_or_noop": invalid, "objective_evaluations": size * len(maxima)},
    )
