"""Varian Hill Climbing sesuai materi IF3170, slide 22 dan 26-29."""

from random import Random
from time import perf_counter

from .common import configuration, initial_state
from ..model import RunResult
from ..neighbors import best_feasible_neighbor, sample_feasible_neighbor
from ..validation import objective


def _result(config, first, current, score, history, started, stop_reason, metrics):
    # Format hasilnya sama buat semua HC, loop pencariannya tetep masing2
    return RunResult(
        algorithm="hc", seed=config["seed"], config=config,
        initial_state=first, final_state=current, best_state=current,
        initial_score=history[0], final_score=score, best_score=score,
        objective_history=tuple(history), duration_seconds=perf_counter() - started,
        stop_reason=stop_reason, metrics=metrics,
    )


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

    return _result(
        config, first, current, score, history, started, stop_reason,
        dict(variant="steepest_ascent", iterations=len(history) - 1,
             accepted_moves=accepted_moves, evaluated_neighbors=evaluated_neighbors),
    )


def sideways(problem, config=None, initial=None):
    # Skor sama boleh diterima, tapi jumlah sideways berturut2 dibatesin
    started = perf_counter()
    config = configuration("hc", {} if config is None else config, "sideways")
    rng = Random(config["seed"])
    first = initial_state(problem, rng, initial)
    current = first
    score = objective(problem, current)
    history = [score]
    accepted_moves, evaluated_neighbors = 0, 0
    sideways_moves, consecutive_sideways = 0, 0
    stop_reason = "iteration_limit"

    for _ in range(config["iterations"]):
        neighbor, neighbor_score, evaluated = best_feasible_neighbor(problem, current, rng)
        evaluated_neighbors += evaluated
        if neighbor is None or neighbor_score < score:
            history.append(score)
            stop_reason = "no_feasible_neighbor" if neighbor is None else "local_optimum"
            break
        if neighbor_score == score:
            if config["max_sideways_moves"] == 0:
                history.append(score)
                stop_reason = "sideways_limit"
                break
            sideways_moves += 1
            consecutive_sideways += 1
        else:
            consecutive_sideways = 0
        current, score = neighbor, neighbor_score
        accepted_moves += 1
        history.append(score)
        if consecutive_sideways > 0 and consecutive_sideways >= config["max_sideways_moves"]:
            stop_reason = "sideways_limit"
            break

    return _result(
        config, first, current, score, history, started, stop_reason,
        dict(variant="sideways", iterations=len(history) - 1,
             accepted_moves=accepted_moves, evaluated_neighbors=evaluated_neighbors,
             sideways_moves=sideways_moves, consecutive_sideways=consecutive_sideways),
    )


def stochastic(problem, config=None, initial=None):
    # Coba satu neighbor random tiap iterasi, pindah cuma kalo value naik
    started = perf_counter()
    config = configuration("hc", {} if config is None else config, "stochastic")
    rng = Random(config["seed"])
    first = initial_state(problem, rng, initial)
    current = first
    score = objective(problem, current)
    history = [score]
    accepted_moves, evaluated_neighbors = 0, 0
    sampling_attempts, sampling_failures = 0, 0

    for _ in range(config["iterations"]):
        sampled = sample_feasible_neighbor(problem, current, rng, config["max_attempts"])
        sampling_attempts += sampled.attempts
        if sampled.state is None:
            sampling_failures += 1
            history.append(score)
            continue
        neighbor_score = objective(problem, sampled.state)
        evaluated_neighbors += 1
        if neighbor_score > score:
            current, score = sampled.state, neighbor_score
            accepted_moves += 1
        history.append(score)

    return _result(
        config, first, current, score, history, started, "iteration_limit",
        dict(variant="stochastic", iterations=len(history) - 1,
             accepted_moves=accepted_moves, evaluated_neighbors=evaluated_neighbors,
             sampling_attempts=sampling_attempts, sampling_failures=sampling_failures),
    )


def random_restart(problem, config=None, initial=None):
    # Ulang steepest ascent dari state random baru, simpen hasil terbaik semua run
    started = perf_counter()
    config = configuration("hc", {} if config is None else config, "random_restart")
    rng = Random(config["seed"])
    first = initial_state(problem, rng, initial)
    best = first
    best_score = objective(problem, first)
    history, runs = [], []
    base_config = {key: config[key] for key in ("seed", "iterations", "max_attempts")}

    for restart in range(config["max_restarts"] + 1):
        start = first if restart == 0 else initial_state(problem, rng, None)
        result = steepest_ascent(problem, base_config, start)
        if result.final_score > best_score:
            best, best_score = result.final_state, result.final_score
        history.extend(result.objective_history)
        runs.append(dict(restart=restart, initial_state=start.to_dict(),
                         final_state=result.final_state.to_dict(),
                         objective_history=list(result.objective_history),
                         stop_reason=result.stop_reason, **result.metrics))

    return _result(
        config, first, best, best_score, history, started, "restart_limit",
        dict(variant="random_restart", restarts=len(runs) - 1, runs=runs,
             iterations=sum(run["iterations"] for run in runs),
             iterations_per_restart=[run["iterations"] for run in runs],
             accepted_moves=sum(run["accepted_moves"] for run in runs),
             evaluated_neighbors=sum(run["evaluated_neighbors"] for run in runs)),
    )


VARIANTS = {
    "steepest_ascent": steepest_ascent,
    "sideways": sideways,
    "stochastic": stochastic,
    "random_restart": random_restart,
}


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
    parser.add_argument("--variant", choices=VARIANTS, default="steepest_ascent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--max-sideways-moves", type=int, default=100)
    parser.add_argument("--max-restarts", type=int, default=3)
    args = parser.parse_args()
    problem = load_problem(args.input)
    options = dict(seed=args.seed, iterations=args.iterations, max_attempts=args.max_attempts)
    extra_options = {
        "sideways": dict(max_sideways_moves=args.max_sideways_moves),
        "random_restart": dict(max_restarts=args.max_restarts),
    }
    options.update(extra_options.get(args.variant, {}))
    result = VARIANTS[args.variant](problem, options)
    # Simpen JSON dan bikin visualisasi setelah pencarian, biar ga masuk durasi run
    directory = args.output / f"{args.variant}_seed{args.seed}"
    directory.mkdir(parents=True, exist_ok=True)
    save_result(directory / "result.json", problem, result, args.input.stem)
    render_run(problem, result, directory, args.input.stem)
    print(f"{args.variant}: {result.initial_score:g} -> {result.final_score:g}")
    print(f"Iterasi: {result.metrics['iterations']} | Stop: {result.stop_reason}")
    print(f"Durasi: {result.duration_seconds:.4f}s | Output: {directory}")
