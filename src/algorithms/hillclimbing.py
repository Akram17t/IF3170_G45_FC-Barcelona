"""Steepest Ascent Hill Climbing sesuai materi IF3170, slide 22."""

from random import Random
from time import perf_counter

from .common import configuration, initial_state
from ..model import RunResult
from ..neighbors import best_feasible_neighbor
from ..validation import objective


def steepest_ascent(problem, config=None, initial=None):
    # Ambil neighbor terbaik tiap iterasi, stop kalo skornya ga naik
    started = perf_counter()
    config = configuration("hc", {} if config is None else config)
    rng = Random(config["seed"])
    first = initial_state(problem, rng, initial)
    current = first
    score = objective(problem, current)
    history = [score]
    accepted_moves, evaluated_neighbors = 0, 0
    stop_reason = "iteration_limit"

    for _ in range(config["iterations"]):
        # Semua neighbor dicek, ambil yang value-nya paling tinggi
        neighbor, neighbor_score, evaluated = best_feasible_neighbor(problem, current, rng)
        evaluated_neighbors += evaluated
        # Kalo yang terbaik aja ga lebih bagus, pencarian berhenti
        if neighbor is None or neighbor_score <= score:
            history.append(score)
            stop_reason = "no_feasible_neighbor" if neighbor is None else "local_optimum"
            break
        current, score = neighbor, neighbor_score
        accepted_moves += 1
        history.append(score)

    return RunResult(
        algorithm="hc", seed=config["seed"], config=config,
        initial_state=first, final_state=current, best_state=current,
        initial_score=history[0], final_score=score, best_score=score,
        objective_history=tuple(history), duration_seconds=perf_counter() - started,
        stop_reason=stop_reason,
        metrics=dict(iterations=len(history) - 1, accepted_moves=accepted_moves,
                     evaluated_neighbors=evaluated_neighbors),
    )


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from ..model import load_problem
    from ..results import save_result
    from ..visualization import render_run

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "examples" / "tiny.json")
    parser.add_argument("--output", type=Path, default=root / "output" / "hc")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()
    problem = load_problem(args.input)
    result = steepest_ascent(problem, dict(seed=args.seed, iterations=args.iterations))
    # Simpen JSON dan bikin visualisasi setelah pencarian, biar ga masuk durasi run
    directory = args.output / f"steepest_ascent_seed{args.seed}"
    directory.mkdir(parents=True, exist_ok=True)
    save_result(directory / "result.json", problem, result, args.input.stem)
    render_run(problem, result, directory, args.input.stem)
    print(f"Steepest ascent: {result.initial_score:g} -> {result.final_score:g}")
    print(f"Iterasi: {result.metrics['iterations']} | Stop: {result.stop_reason}")
    print(f"Durasi: {result.duration_seconds:.4f}s | Output: {directory}")
