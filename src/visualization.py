"""Grafik statis headless. API plot: https://matplotlib.org/stable/users/index.html"""

import math
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .model import State, occupancy, oriented_size
from .validation import objective, total_weight


# Palet FC Barcelona: blau, grana, vermell, groc, groc clar.
BLUE, GARNET, RED, GOLD, YELLOW = "#004D98", "#A50044", "#DB0030", "#EDBB00", "#FFED02"
PALETTE = [BLUE, GARNET, GOLD, RED, YELLOW]
DARK = {BLUE, GARNET, RED}


def _tint(color, amount=0.5):
    """Campur warna hex dengan putih agar paket ke-6 dst. tetap berbeda."""
    rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(c + (255 - c) * amount):02X}" for c in rgb)


def package_color(index):
    return PALETTE[index % 5] if index % 10 < 5 else _tint(PALETTE[index % 5])


plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "savefig.facecolor": "white"})


def plot_state(problem, state, path, title):
    cells = occupancy(problem, state)
    count = len(problem.packages)
    colors = ["#f0f3f7"] + [package_color(i) for i in range(count)]
    ids = {p.id: i + 1 for i, p in enumerate(problem.packages)}
    columns = min(3, problem.truck.h)
    rows = math.ceil(problem.truck.h / columns)
    table_height = max(1.7, 0.25 * (count + 1))
    figure = plt.figure(figsize=(12, 3.1 * rows + table_height + 0.9), layout="constrained")
    grid = figure.add_gridspec(rows + 1, columns, height_ratios=[3.1] * rows + [table_height])
    figure.suptitle(f"{title}\nValue {objective(problem, state):g} | "
                   f"Berat {total_weight(problem, state):g}/{problem.truck.max_capacity:g} | "
                   "titik kosong = sel kosong", fontsize=12)
    for z in range(problem.truck.h):
        axis = figure.add_subplot(grid[z // columns, z % columns])
        data = [[ids.get(cells[x, y, z], 0) for x in range(problem.truck.w)]
                for y in range(problem.truck.l)]
        axis.imshow(data, origin="lower", cmap=ListedColormap(colors), vmin=0, vmax=max(1, count))
        for y in range(problem.truck.l):
            for x in range(problem.truck.w):
                index = ids.get(cells[x, y, z], 0) - 1
                text_color = "white" if index >= 0 and package_color(index) in DARK else "black"
                axis.text(x, y, cells[x, y, z] or ".", ha="center", va="center", fontsize=8, color=text_color)
        axis.set(xticks=range(problem.truck.w), yticks=range(problem.truck.l),
                 xlabel="x / width", ylabel="y / length", title=f"Layer z={z}")
        axis.set_xticks([i - 0.5 for i in range(problem.truck.w + 1)], minor=True)
        axis.set_yticks([i - 0.5 for i in range(problem.truck.l + 1)], minor=True)
        axis.grid(which="minor", color="white", linewidth=1.4)
        axis.tick_params(which="minor", length=0)
    table_axis = figure.add_subplot(grid[-1, :])
    table_axis.axis("off")
    table_rows = []
    for package in problem.packages:
        place = state.placements[package.id]
        table_rows.append([package.id, "luar" if place.position is None else "dalam",
                           str(place.position) if place.position is not None else "-",
                           place.orientation.value, str(oriented_size(package, place.orientation)),
                           f"{package.weight:g}", f"{package.value:g}",
                           "ya" if package.is_fragile else "tidak", str(package.eta)])
    if table_rows:
        table = table_axis.table(cellText=table_rows,
                                 colLabels=["ID", "Status", "(x,y,z)", "Orientasi", "(w,l,h)",
                                            "Berat", "Value", "Fragile", "ETA"],
                                 loc="center", cellLoc="center", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#dde3eb")
            if row == 0:
                cell.set_facecolor(BLUE)
                cell.set_text_props(color="white", weight="bold")
    else:
        table_axis.text(0.5, 0.5, "Tidak ada paket", ha="center")
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def plot_curves(result, directory, title):
    figure, axis = plt.subplots(figsize=(9, 4), layout="constrained")
    history = result.objective_history
    axis.plot(range(len(history)), history, label="Maksimum populasi" if result.algorithm == "ga" else "Current value", color=BLUE)
    if result.algorithm == "ga":
        axis.plot(result.metrics["mean_history"], label="Rata-rata populasi", color=GARNET)
    if result.algorithm == "sa":
        axis.plot([result.initial_score] + [r["best_objective"] for r in result.metrics["records"]],
                  label="Best-so-far", color=GOLD, linewidth=2, linestyle="--")
    axis.set(title=title, xlabel="Generasi" if result.algorithm == "ga" else "Iterasi", ylabel="Objective (total value)")
    axis.set_title(title, fontsize=10)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(Path(directory) / "objective.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    if result.algorithm == "sa":
        records = result.metrics["records"]
        cap = 1e6
        ratios = [min(cap, r["exp_delta_over_t"]) if r["exp_delta_over_t"] is not None else cap for r in records]
        figure, axis = plt.subplots(figsize=(9, 4), layout="constrained")
        axis.plot([r["iteration"] for r in records], ratios, color=RED, linewidth=1)
        axis.axhline(1, color="gray", linestyle="--", label="rasio = 1")
        clipped = sum(r["exp_delta_over_t"] is None or r["exp_delta_over_t"] > cap for r in records)
        axis.set(yscale="symlog", title=f"{title}\nexp(delta/T); batas gambar 1e6 ({clipped} titik dibatasi)",
                 xlabel="Iterasi", ylabel="exp(delta/T), skala symlog")
        axis.title.set_fontsize(10)
        axis.grid(alpha=0.2)
        axis.legend()
        figure.savefig(Path(directory) / "acceptance.png", dpi=150, bbox_inches="tight")
        plt.close(figure)


def render_run(problem, result, directory, case_name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    params = ", ".join(f"{k}={v}" for k, v in result.config.items() if k != "seed")
    title = f"{case_name} | {result.algorithm.upper()} | seed={result.seed}\n" + textwrap.fill(params, 78)
    plot_state(problem, result.initial_state, directory / "initial.png", f"Initial | {title}")
    plot_state(problem, result.final_state, directory / "final.png", f"Final | {title}")
    plot_curves(result, directory, title)
    if result.algorithm == "sa":
        plot_state(problem, State.from_dict(problem, result.metrics["final_current"]),
                   directory / "final_current.png", f"Last current | {title}")
