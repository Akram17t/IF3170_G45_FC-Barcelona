"""Visualisasi state dengan Plotly dan grafik eksperimen dengan Matplotlib."""

from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .model import oriented_size
from .validation import objective

# Warna Palet Barcelona
PALETTE = ["#004D98", "#A50044", "#EDBB00", "#DB0030", "#FFED02"]


# Geometri balok dari contoh Mesh Cube: https://plotly.com/python/3d-mesh/
VERTICES = [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0),
            (0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)]
TRIANGLES = ([7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
             [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
             [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6])
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def _vertices(position, size):
    return [[position[axis] + vertex[axis] * size[axis] for vertex in VERTICES]
            for axis in range(3)]


def plot_state(problem, state, path, title="State"):
    """Paket tetap di tempat; drag menggerakkan kamera bawaan Plotly."""
    score = objective(problem, state)
    figure = go.Figure()
    truck = _vertices((0, 0, 0), problem.truck.dimensions)
    lines = [[value for a, b in EDGES for value in (axis[a], axis[b], None)] for axis in truck]
    figure.add_trace(go.Scatter3d(x=lines[0], y=lines[1], z=lines[2], mode="lines",
                                 line=dict(color="gray", width=3), name="Truk", hoverinfo="skip"))
    outside_x = problem.truck.w + 1
    for index, package in enumerate(problem.packages):
        placement = state.placements[package.id]
        size = oriented_size(package, placement.orientation)
        outside = placement.position is None
        # Posisi di samping truk hanya untuk menampilkan paket luar, bukan mengubah state.
        position = (outside_x, 0, 0) if outside else placement.position
        if outside:
            outside_x += size[0] + 1
        x, y, z = _vertices(position, size)
        name = escape(package.id) + (" (luar)" if outside else "")
        figure.add_trace(go.Mesh3d(
            x=x, y=y, z=z, i=TRIANGLES[0], j=TRIANGLES[1], k=TRIANGLES[2],
            color=PALETTE[index % len(PALETTE)],
            name=name, showlegend=True, flatshading=True,
            text=f"{name}<br>Posisi: {placement.position}<br>Orientasi: {placement.orientation.value}",
            hoverinfo="text"))
    figure.update_layout(
        title=f"{escape(title)} | Value: {score:g}",
        scene=dict(aspectmode="data", dragmode="turntable",
                   camera=dict(up=dict(x=0, y=0, z=1)),
                   xaxis_title="x", yaxis_title="y", zaxis_title="z"),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs=True, config=dict(scrollZoom=True, displaylogo=False))
    return figure


def plot_curves(result, directory, title):
    figure, axis = plt.subplots()
    if result.algorithm == "hc" and result.metrics.get("variant") == "random_restart":
        # Tiap restart diplot sendiri biar iterasinya ga kecampur
        for index, run in enumerate(result.metrics["runs"]):
            axis.plot(run["objective_history"], color=PALETTE[index % len(PALETTE)],
                      label=f"Run {index}", marker=".")
    else:
        axis.plot(result.objective_history, color=PALETTE[0],
                  label="Max populasi" if result.algorithm == "ga" else "Value")
    if result.algorithm == "ga":
        axis.plot(result.metrics["mean_history"], color=PALETTE[1], label="Rata-rata populasi")
    axis.set(title=title, xlabel="Generasi" if result.algorithm == "ga" else "Iterasi", ylabel="Objective")
    axis.legend()
    figure.savefig(Path(directory) / "objective.png", bbox_inches="tight")
    plt.close(figure)
    if result.algorithm == "sa":
        records = result.metrics["records"]
        ratios = [min(1e6, r["exp_delta_over_t"]) if r["exp_delta_over_t"] is not None else 1e6
                  for r in records]
        figure, axis = plt.subplots()
        axis.plot([r["iteration"] for r in records], ratios, color=PALETTE[3])
        axis.set(title=title, xlabel="Iterasi", ylabel="exp(delta/T), dibatasi 1e6", yscale="symlog")
        figure.savefig(Path(directory) / "acceptance.png", bbox_inches="tight")
        plt.close(figure)


def render_run(problem, result, directory, case_name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    algorithm = result.metrics.get("variant", result.algorithm.upper())
    title = f"{case_name} | {algorithm} | seed={result.seed}"
    plot_state(problem, result.initial_state, directory / "initial.html", f"Initial | {title}")
    plot_state(problem, result.final_state, directory / "final.html", f"Final | {title}")
    plot_curves(result, directory, title)


if __name__ == "__main__":
    from random import Random
    from .initialize import random_feasible_state
    from .model import load_problem

    root = Path(__file__).resolve().parents[1]
    problem = load_problem(root / "examples" / "tiny.json")
    path = root / "output" / "initial.html"
    plot_state(problem, random_feasible_state(problem, Random(0)), path, "Initial")
    print(path)
