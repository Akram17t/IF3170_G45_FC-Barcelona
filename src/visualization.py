from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
        ratios = [min(1e6, r["exp_delta_over_t"]) if r["exp_delta_over_t"] is not None else 1e6
                  for r in records]
        figure, axis = plt.subplots()
        axis.plot([r["iteration"] for r in records], ratios, color=PALETTE[3])
        axis.set(title=title, xlabel="Iterasi", ylabel="exp(delta/T), dibatasi 1e6", yscale="symlog")
        figure.savefig(Path(directory) / "acceptance.png", bbox_inches="tight")
        plt.close(figure)


def render_run(problem, result, directory, case_name):
    # problem tidak dipakai lagi, tapi dipertahankan biar pemanggil di HC/SA/GA tidak berubah
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    algorithm = result.metrics.get("variant", result.algorithm.upper())
    title = f"{case_name} | {algorithm} | seed={result.seed}"
    plot_curves(result, directory, title)
