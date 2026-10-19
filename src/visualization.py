from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .model import oriented_size
from .validation import objective

# Warna Palet Barca
PALETTE = ["#004D98", "#A50044", "#EDBB00", "#DB0030", "#FFED02"]


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
        # e^(dE/T) >= 1 berarti move membaik (selalu diterima), jadi dibatasi 1 biar peluang move buruk kelihatan
        ratios = [min(1.0, r["exp_delta_over_t"]) if r["exp_delta_over_t"] is not None else 1.0
                  for r in records]
        figure, axis = plt.subplots()
        axis.plot([r["iteration"] for r in records], ratios, color=PALETTE[3], marker=".", markersize=3, linewidth=0.5)
        axis.set(title=title, xlabel="Iterasi", ylabel="min(1, exp(delta/T))", ylim=(-0.05, 1.05))
        figure.savefig(Path(directory) / "acceptance.png", bbox_inches="tight")
        plt.close(figure)


def box_faces(x, y, z, w, l, h):
    # Enam sisi balok dari sudut kiri-bawah-belakang (x, y, z)
    c = [(x + i * w, y + j * l, z + k * h) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    return [[c[0], c[1], c[3], c[2]], [c[4], c[5], c[7], c[6]], [c[0], c[1], c[5], c[4]],
            [c[2], c[3], c[7], c[6]], [c[0], c[2], c[6], c[4]], [c[1], c[3], c[7], c[5]]]


def outside_layout(truck, boxes, widen):
    # Paket di luar truk disusun berderet (rak sederhana). Kalau widen, zona dilebarkan sampai panjangnya
    # tidak melebihi truk, atau sampai tiga kali lebar truk.
    limits = (truck.w, truck.w * 3 // 2, truck.w * 2, truck.w * 3) if widen else (truck.w,)
    for limit in limits:
        placed, x, y, row = [], 0, 0, 0
        for color, w, l, h in boxes:
            if x > 0 and x + w > limit:
                x, y, row = 0, y + row + 1, 0
            placed.append((x, y, 0, w, l, h, color))
            x, row = x + w + 1, max(row, l)
        if y + row <= truck.l:
            break
    return placed, limit, y + row


def state_boxes(problem, state, below):
    # Kuning = fragile, biru/garnet = paket biasa (bergantian per paket supaya batasnya terlihat).
    # Paket di luar truk disusun di zona di bawah truk (sumbu y, untuk gambar vertikal) atau di sebelah kanannya.
    truck = problem.truck
    inside, outside = [], []
    for index, package in enumerate(problem.packages):
        placement = state.placements[package.id]
        color = PALETTE[2] if package.is_fragile else PALETTE[index % 2]
        w, l, h = oriented_size(package, placement.orientation)
        if placement.position is None:
            outside.append((color, w, l, h))
        else:
            inside.append((*placement.position, w, l, h, color))
    placed, zone_width, zone_length = outside_layout(truck, outside, widen=not below)
    ox, oy = (0, truck.l + 4) if below else (truck.w + 4, 0)
    moved = [(x + ox, y + oy, z, w, l, h, color) for x, y, z, w, l, h, color in placed]
    # Paket yang lebih tinggi digambar belakangan supaya tidak tertutup
    boxes = sorted(inside + moved, key=lambda box: box[2])
    zone = (ox, oy, ox + zone_width, oy + max(zone_length, 0 if below else truck.l))
    return boxes, zone, max(truck.w, zone[2]), max(truck.l, zone[3])


def plot_state(problem, state, path):
    # Gambar 3D vertikal (portrait): truk di bawah, paket di luar truk di zona garis putus-putus di atasnya
    truck = problem.truck
    boxes, zone, width, length = state_boxes(problem, state, below=True)
    figure = plt.figure(figsize=(7, 11))
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axis = figure.add_subplot(projection="3d")
    axis.add_collection3d(Poly3DCollection(box_faces(0, 0, 0, truck.w, truck.l, truck.h),
                                           facecolors="none", edgecolors="#444444", linewidths=1.2))
    # Zona paket di luar truk hanya garis putus-putus di lantai, supaya tidak menutupi paket
    x0, y0, x1, y1 = zone
    axis.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], 0, color="#999999", linestyle="--")
    for x, y, z, w, l, h, color in boxes:
        axis.add_collection3d(Poly3DCollection(box_faces(x, y, z, w, l, h), facecolors=color,
                                               edgecolors="white", linewidths=0.6, alpha=0.65))
    axis.set(xlim=(0, width), ylim=(0, length), zlim=(0, truck.h))
    axis.set_box_aspect((width, length, truck.h), zoom=1.2)
    axis.view_init(elev=38, azim=-72)
    # Sumbu digambar manual di tepi truk saja (zona paket di luar tidak berskala), angka tiap 2 satuan
    axis.set_axis_off()
    for i in range(0, truck.w + 1, 2):
        axis.text(i, -0.8, 0, str(i), ha="center", va="top", fontsize=11)
    for j in range(0, truck.l + 1, 2):
        axis.text(truck.w + 0.8, j, 0, str(j), ha="left", va="center", fontsize=11)
    for k in range(truck.h + 1):
        axis.text(-0.8, 0, k, str(k), ha="right", va="center", fontsize=11)
    axis.text(truck.w / 2, -3, 0, "x", ha="center", va="top", fontsize=14)
    axis.text(truck.w + 3, truck.l / 2, 0, "y", ha="left", va="center", fontsize=14)
    axis.text(-2.5, 0, truck.h / 2, "z", ha="right", va="center", fontsize=14)
    figure.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(figure)


def plot_top(problem, state, path):
    # Tampak atas (sumbu x dan y), zona paket di luar truk di sebelah kanan
    truck = problem.truck
    boxes, zone, width, length = state_boxes(problem, state, below=False)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.add_patch(Rectangle((0, 0), truck.w, truck.l, fill=False, edgecolor="#444444", linewidth=1.5))
    axis.add_patch(Rectangle((zone[0], zone[1]), zone[2] - zone[0], zone[3] - zone[1], facecolor="#EEEEEE",
                             edgecolor="#999999", linestyle="--"))
    for x, y, z, w, l, h, color in boxes:
        axis.add_patch(Rectangle((x, y), w, l, facecolor=color, edgecolor="white", alpha=0.8))
    axis.set(xlim=(0, width), ylim=(0, length), xticks=range(0, truck.w + 1, 2),
             yticks=range(0, truck.l + 1, 2), xlabel="x", ylabel="y")
    axis.set_aspect("equal")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    figure.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(figure)


def render_run(problem, result, directory, case_name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    algorithm = result.metrics.get("variant", result.algorithm.upper())
    title = f"{case_name} | {algorithm} | seed={result.seed}"
    plot_curves(result, directory, title)
    for label, state in (("awal", result.initial_state), ("akhir", result.final_state)):
        plot_state(problem, state, directory / f"{label}.png")
        plot_top(problem, state, directory / f"{label}_atas.png")
