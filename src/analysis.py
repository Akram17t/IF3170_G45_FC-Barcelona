"""Analisis dari JSON run, serta batas atas relaksasi berat/volume."""

from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean, pstdev


def _knapsack_bound(values, costs, capacity):
    if capacity <= 10000 and all(float(cost).is_integer() for cost in costs):
        limit = int(capacity)
        scores = [0.0] * (limit + 1)
        for value, cost in zip(values, costs):
            cost = int(cost)
            for available in range(limit, cost - 1, -1):
                scores[available] = max(scores[available], scores[available - cost] + value)
        return scores[-1]
    # Relaksasi fractional juga merupakan batas atas untuk pemilihan 0/1.
    result = sum(value for value, cost in zip(values, costs) if cost == 0)
    ordered = sorted(((value / cost, value, cost) for value, cost in zip(values, costs) if cost > 0), reverse=True)
    remaining = capacity
    for _, value, cost in ordered:
        fraction = min(1.0, remaining / cost)
        result += fraction * value
        remaining -= fraction * cost
        if remaining <= 0:
            break
    return result


def value_upper_bound(problem):
    values = [p.value for p in problem.packages]
    weight_bound = _knapsack_bound(values, [p.weight for p in problem.packages], problem.truck.max_capacity)
    volume_bound = _knapsack_bound(values, [math.prod(p.dimensions) for p in problem.packages], math.prod(problem.truck.dimensions))
    return {"value": min(sum(values), weight_bound, volume_bound), "weight_relaxation": weight_bound,
            "volume_relaxation": volume_bound,
            "definition": "Minimum batas atas total value, knapsack berat, dan knapsack volume; packing/support diabaikan."}


def build_analysis(problem, rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["group"], row["population_size"], row["generations"])].append(row)
    summaries = []
    for (group, population, generations), items in groups.items():
        scores = [row["final_score"] for row in items]
        durations = [row["duration_seconds"] for row in items]
        summaries.append({"group": group, "population_size": population, "generations": generations,
                          "runs": len(items), "seeds": [row["seed"] for row in items],
                          "score_mean": mean(scores), "score_min": min(scores), "score_max": max(scores),
                          "score_std_population": pstdev(scores), "duration_mean": mean(durations),
                          "duration_min": min(durations), "duration_max": max(durations)})
    bound = value_upper_bound(problem)
    best = max(row["final_score"] for row in rows)
    return {"upper_bound": bound, "best_observed": best,
            "bound_attained": math.isclose(best, bound["value"], rel_tol=1e-10, abs_tol=1e-10),
            "groups": summaries}


def write_analysis(problem, rows, directory, plots=True):
    directory = Path(directory)
    analysis = build_analysis(problem, rows)
    (directory / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not plots:
        return analysis
    from .visualization import BLUE, GARNET, GOLD, plt
    groups = analysis["groups"]
    hc, sa = [next(g for g in groups if g["group"] == name) for name in ("hc", "sa")]
    # Gunakan baseline GA yang sama pada dua sweep, hanya dihitung sekali.
    population_sweep = [g for g in groups if g["group"] == "ga_population"]
    fixed_g = population_sweep[0]["generations"]
    generation_sweep = [g for g in groups if g["group"] == "ga_generations"]
    fixed_p = generation_sweep[0]["population_size"]
    baseline = next((g for g in generation_sweep if g["generations"] == fixed_g), generation_sweep[0])
    comparison = [hc, sa, baseline]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    labels = ["HC", "SA", f"GA P={fixed_p}\nG={baseline['generations']}"]
    for axis, field, low, high, ylabel in (
        (axes[0], "score_mean", "score_min", "score_max", "Final objective"),
        (axes[1], "duration_mean", "duration_min", "duration_max", "Durasi pencarian (detik)"),
    ):
        values = [g[field] for g in comparison]
        axis.bar(labels, values, color=[BLUE, GARNET, GOLD])
        axis.errorbar(range(3), values,
                      yerr=[[g[field] - g[low] for g in comparison], [g[high] - g[field] for g in comparison]],
                      fmt="none", ecolor="#172a40", capsize=5)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=.2)
    axes[0].axhline(analysis["upper_bound"]["value"], color="gray", linestyle="--", label="Batas atas")
    axes[0].legend()
    figure.suptitle("Benchmark: rata-rata tiga seed; error bar = minimum hingga maksimum")
    figure.savefig(directory / "comparison.png", dpi=150)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, group, field, title in ((axes[0], generation_sweep, "generations", f"P tetap = {fixed_p}"),
                                      (axes[1], population_sweep, "population_size", f"G tetap = {fixed_g}")):
        group = sorted(group, key=lambda g: g[field])
        axis.errorbar([g[field] for g in group], [g["score_mean"] for g in group],
                      yerr=[[g["score_mean"] - g["score_min"] for g in group], [g["score_max"] - g["score_mean"] for g in group]],
                      marker="o", capsize=5, color=BLUE)
        axis.set(title=title, xlabel="Generasi" if field == "generations" else "Populasi", ylabel="Final objective")
        axis.grid(alpha=.2)
    figure.suptitle("Dua sweep GA: mean dan rentang tiga seed")
    figure.savefig(directory / "ga_sweeps.png", dpi=150)
    plt.close(figure)
    return analysis
