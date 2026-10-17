from math import fsum
from random import Random
from time import perf_counter

from .common import configuration, initial_state
from ..initialize import random_feasible_state
from ..model import Placement, RunResult, State
from ..neighbors import sample_feasible_neighbor
from ..validation import objective, validate


def roulette(population, scores, rng):
    # Peluang kepilih sebanding skor, kalo semua skor 0 pilih acak
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
    # Satu titik potong di urutan paket, hasilnya dua anak
    ids = []
    
    for package in problem.packages:
        ids.append(package.id)
    
    cut = 0
    
    if len(ids) > 1:
        cut = rng.randrange(1, len(ids))
    
    children = []
    
    # Anak 1 = depan dari first + belakang dari second, anak 2 sebaliknya
    for a, b in ((first, second), (second, first)):
        placements = {}
        for index, package_id in enumerate(ids):
            if index < cut:
                placements[package_id] = a.placements[package_id]
            else:
                placements[package_id] = b.placements[package_id]
        children.append(State(placements))
    
    return children


def repair(problem, child, rng):
    # Paket diurut dari lantai ke atas, yang bikin state ga valid dikeluarin
    # Mulai dari semua paket di luar, orientasinya ngikutin child
    placements = {}
    ids = []
    
    for package in problem.packages:
        placements[package.id] = Placement(None, child.placements[package.id].orientation)
        if child.placements[package.id].position is not None:
            ids.append(package.id)
    
    state = State(placements)
    rng.shuffle(ids)  # urutan acak buat paket yang z-nya sama
    
    # Urut dari lantai (z=0) ke atas
    ordered = []
    for z in range(problem.truck.h):
        for package_id in ids:
            if child.placements[package_id].position[2] == z:
                ordered.append(package_id)
    
    for package_id in ordered:
        placements = dict(state.placements)
        placements[package_id] = child.placements[package_id]
        candidate = State(placements)
        if validate(problem, candidate).valid:
            state = candidate
    
    return state


def mutate(problem, state, rng, max_attempts):
    # Satu move acak, kalo ga ketemu yang valid state-nya tetap
    neighbor = sample_feasible_neighbor(problem, state, rng, max_attempts).state
    
    if neighbor is None:
        return state
    
    return neighbor


def _result(config, first, best, best_score, history, means, started, first_population,
            first_scores, population, scores):
    size = config["population_size"]
    
    # Ukuran populasi dicatat tiap generasi, State diubah jadi dict biar bisa masuk JSON
    population_sizes = []
    for _ in history:
        population_sizes.append(size)
    
    initial_population = []
    for state in first_population:
        initial_population.append(state.to_dict())
    
    final_population = []
    for state in population:
        final_population.append(state.to_dict())
    
    return RunResult(
        algorithm="ga", seed=config["seed"], config=config,
        initial_state=first, final_state=population[scores.index(max(scores))], best_state=best,
        initial_score=history[0], final_score=history[-1], best_score=best_score,
        objective_history=tuple(history), duration_seconds=perf_counter() - started,
        stop_reason="generation_limit",
        metrics=dict(
            max_history=list(history), mean_history=means, population_sizes=population_sizes,
            evaluations=size * len(history),
            initial_population=initial_population, initial_population_scores=first_scores,
            final_population=final_population, final_population_scores=scores,
        ),
    )


def genetic_algorithm(problem, config=None, initial=None):
    # Populasi diganti satu generasi penuh, kecuali elit yang disalin apa adanya
    started = perf_counter()
    
    if config is None:
        config = {}
    
    config = configuration("ga", config)
    rng = Random(config["seed"])
    size = config["population_size"]
    
    # Individu pertama boleh state yang dikasih, sisanya acak
    population = [initial_state(problem, rng, initial)]
    for _ in range(size - 1):
        population.append(random_feasible_state(problem, rng))
    
    scores = []
    for state in population:
        scores.append(objective(problem, state))
    
    first_population, first_scores = population, scores
    top = scores.index(max(scores))
    first, best, best_score = population[top], population[top], scores[top]
    history = [scores[top]]
    means = [fsum(scores) / size]

    for _ in range(config["generations"]):
        # Elit: ambil individu skor tertinggi sebanyak elitism, disalin apa adanya
        following = []
        left = list(range(size))
    
        for _ in range(config["elitism"]):
            elite = left[0]
    
            for index in left:
                if scores[index] > scores[elite]:
                    elite = index
    
            following.append(population[elite])
            left.remove(elite)
    
        while len(following) < size:
            parent1 = roulette(population, scores, rng)
            parent2 = roulette(population, scores, rng)
    
            for child in crossover(problem, parent1, parent2, rng):
                child = repair(problem, child, rng)
    
                if rng.random() < config["mutation_rate"]:
                    child = mutate(problem, child, rng, config["max_attempts"])
    
                following.append(child)
    
        population = following[:size]  # anak lebihan dibuang kalo ukurannya ganjil
        scores = []
    
        for state in population:
            scores.append(objective(problem, state))
    
        top = scores.index(max(scores))
    
        if scores[top] > best_score:
            best, best_score = population[top], scores[top]
    
        history.append(scores[top])
        means.append(fsum(scores) / size)

    return _result(config, first, best, best_score, history, means, started,
                   first_population, first_scores, population, scores)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from ..model import load_problem
    from ..results import save_result
    from ..visualization import render_run

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "examples" / "tiny.json")
    parser.add_argument("--output", type=Path, default=root / "output" / "ga")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--mutation-rate", type=float, default=0.3)
    parser.add_argument("--elitism", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=100)
    args = parser.parse_args()
    problem = load_problem(args.input)
    result = genetic_algorithm(problem, dict(
        seed=args.seed, generations=args.generations, population_size=args.population_size,
        mutation_rate=args.mutation_rate, elitism=args.elitism, max_attempts=args.max_attempts))
    # Simpen JSON dan bikin visualisasi setelah pencarian
    directory = args.output / f"seed{args.seed}"
    directory.mkdir(parents=True, exist_ok=True)
    save_result(directory / "result.json", problem, result, args.input.stem)
    render_run(problem, result, directory, args.input.stem)
    print(f"ga: {result.initial_score:g} -> {result.final_score:g}")
    print(f"Generasi: {args.generations} | Populasi: {args.population_size} | Stop: {result.stop_reason}")
    print(f"Durasi: {result.duration_seconds:.4f}s | Output: {directory}")
