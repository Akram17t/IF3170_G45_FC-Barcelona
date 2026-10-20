import hashlib
import json
import math
import platform
import sys
from pathlib import Path

from .model import Problem, RunResult, State, occupancy
from .validation import objective


# memastikan semua hasil run memang berasal dari soal (input) yang sama, dan isi soal di file hasil tidak diubah-ubah.
def problem_hash(problem):
    return hashlib.sha256(json.dumps(problem.to_dict(), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sa_directory(root, problem, case_name, seed):
    # nama kasus cuma jadi nama folder
    name = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(case_name))[:64]
    name = name or "custom"
    parent = Path(root) / f"{name}_{problem_hash(problem)[:12]}" / f"seed{seed}"
    parent.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        directory = parent / f"run{number}"
        try:
            # mkdir ini sekalian reservasi folder kalau ada dua request bersamaan
            directory.mkdir()
            return directory
        except FileExistsError:
            number += 1


# biar angka yang cuma beda di 10 digit belakang koma tetap dianggap sama
def _same(a, b, abs_tol=1e-10):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=abs_tol)


# cek skor, histori, dan metrik HC/SA/GA masuk akal, kalau aneh langsung error
def check_result(problem: Problem, result: RunResult):
    from .algorithms.common import configuration

    config = configuration(result.algorithm, result.config, result.metrics.get("variant"))
    if config["seed"] != result.seed:
        raise ValueError("Seed hasil dan konfigurasi berbeda")
    for label in ("initial", "final", "best"):
        state = getattr(result, label + "_state")
        score = getattr(result, label + "_score")
        if state is None or score is None:
            if label != "best" or state is not None or score is not None:
                raise ValueError(f"State/skor {label} tidak lengkap")
        elif not _same(objective(problem, state, config["objective"]), score):
            raise ValueError(f"Skor {label} tidak sesuai state")
    history = result.objective_history
    if not history or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in history):
        raise ValueError("Histori objective tidak sah")
    if not _same(history[0], result.initial_score):
        raise ValueError("Histori harus mulai dari skor initial")
    if not math.isfinite(result.duration_seconds) or result.duration_seconds < 0:
        raise ValueError("Durasi tidak sah")
    if result.best_score is None or not _same(result.best_score, max(history)):
        raise ValueError("Best score tidak sesuai histori")
    metrics = result.metrics
    if result.algorithm == "hc":
        if metrics.get("variant") == "random_restart":
            # Skor boleh turun pas restart, tapi tiap run tetep ga boleh turun
            runs = metrics["runs"]
            if len(runs) != config["max_restarts"] + 1 or metrics["restarts"] != len(runs) - 1:
                raise ValueError("Jumlah restart HC tidak konsisten")
            combined = []
            for index, run in enumerate(runs):
                local = run["objective_history"]
                if (not local or run["restart"] != index or run["iterations"] != len(local) - 1
                        or any(a > b for a, b in zip(local, local[1:]))):
                    raise ValueError("Histori per restart HC tidak konsisten")
                for label, score in (("initial", local[0]), ("final", local[-1])):
                    state = State.from_dict(problem, run[label + "_state"])
                    if not _same(objective(problem, state, config["objective"]), score):
                        raise ValueError("Skor per restart HC tidak sesuai state")
                    if index == 0 and label == "initial" and state != result.initial_state:
                        raise ValueError("Initial state restart pertama berbeda")
                combined.extend(local)
            counts = [run["iterations"] for run in runs]
            if (combined != list(history) or metrics["iterations"] != sum(counts)
                    or metrics["iterations_per_restart"] != counts):
                raise ValueError("Histori gabungan restart HC tidak konsisten")
            if not _same(result.final_score, max(history)):
                raise ValueError("Hasil random restart HC harus yang terbaik")
        else:
            if metrics["iterations"] != len(history) - 1 or any(a > b for a, b in zip(history, history[1:])):
                raise ValueError("Histori HC tidak konsisten")
            if not _same(history[-1], result.final_score):
                raise ValueError("Skor akhir HC tidak sesuai histori")
    elif result.algorithm == "sa":
        from .algorithms.sa import acceptance_probability, temperature_schedule
        from .neighbors import improving_neighbor_exists

        records = metrics["records"]
        iterations = len(records)
        if iterations != len(history) - 1 or metrics["iterations"] != iterations or iterations > config["iterations"]:
            raise ValueError("Histori SA tidak konsisten")
        current = State.from_dict(problem, metrics["final_current"])
        if current != result.final_state or not _same(objective(problem, current, config["objective"]), history[-1]):
            raise ValueError("Current SA harus sama dengan state akhir")
        if not _same(metrics["final_current_score"], history[-1]):
            raise ValueError("Metrik skor current SA tidak sesuai")
        if not _same(result.final_score, history[-1]):
            raise ValueError("Skor akhir SA harus sesuai current terakhir")
        stuck_history = metrics["stuck_history"]
        if (len(stuck_history) != iterations + 1 or any(type(flag) is not bool for flag in stuck_history)
                or type(metrics["initial_stuck"]) is not bool or metrics["initial_stuck"] != stuck_history[0]):
            raise ValueError("Histori stuck SA tidak konsisten")
        if (stuck_history[0] != (not improving_neighbor_exists(problem, result.initial_state, config["objective"]))
                or stuck_history[-1] != (not improving_neighbor_exists(problem, current, config["objective"]))):
            raise ValueError("Stuck SA tidak sesuai state awal/akhir")

        for i, record in enumerate(records, 1):
            if type(record["accepted"]) is not bool or type(record["stuck"]) is not bool:
                raise ValueError("Flag SA harus boolean")
            temperature = temperature_schedule(i, config)
            if temperature == 0 or not _same(record["temperature"], temperature, abs_tol=0):
                raise ValueError("Temperatur SA tidak sesuai schedule")
            expected = history[i - 1] + record["delta"] if record["accepted"] else history[i - 1]
            if record["iteration"] != i or not _same(expected, history[i]):
                raise ValueError("Transisi SA tidak konsisten")
            if not _same(record["objective"], history[i]):
                raise ValueError("Record SA tidak sesuai histori")
            probability = acceptance_probability(record["delta"], record["temperature"])
            # Peluang sangat kecil jangan dianggap sama dengan nol karena toleransi absolut
            if not _same(probability, record["acceptance_probability"], abs_tol=0):
                raise ValueError("Probabilitas SA tidak sesuai delta/T")
            if (probability == 1 and not record["accepted"]) or (probability == 0 and record["accepted"]):
                raise ValueError("Keputusan penerimaan SA tidak mungkin")
            if record["stuck"] != stuck_history[i] or (not record["accepted"] and stuck_history[i] != stuck_history[i - 1]):
                raise ValueError("Record stuck SA tidak sesuai histori")
            if record["delta"] > 0 and stuck_history[i - 1]:
                raise ValueError("State stuck SA tidak boleh punya kandidat lebih baik")
            log_ratio = record["delta"] / record["temperature"]
            ratio = record["exp_delta_over_t"]
            if type(record["log_ratio_overflow"]) is not bool or record["log_ratio_overflow"] != (not math.isfinite(log_ratio)):
                raise ValueError("Flag overflow log SA tidak sesuai")
            if not math.isfinite(log_ratio) and record["log_acceptance_ratio"] is not None:
                raise ValueError("Log ratio overflow SA harus null")
            if math.isfinite(log_ratio) and not _same(record["log_acceptance_ratio"], log_ratio, abs_tol=0):
                raise ValueError("Log ratio SA tidak sesuai")
            if log_ratio <= math.log(sys.float_info.max):
                if ratio is None or not _same(ratio, math.exp(log_ratio), abs_tol=0) or record["exp_overflow"]:
                    raise ValueError("Exp SA tidak sesuai delta/T")
            elif ratio is not None or not record["exp_overflow"]:
                raise ValueError("Overflow exp SA tidak dicatat")
            if type(record["exp_overflow"]) is not bool:
                raise ValueError("Flag overflow exp SA harus boolean")

        counts = {
            "iterations": iterations,
            "accepted_moves": sum(r["accepted"] for r in records),
            "accepted_moves_worse": sum(r["accepted"] and r["delta"] < 0 for r in records),
            "rejected_moves": sum(not r["accepted"] for r in records),
            "stuck_i": sum(stuck_history[1:]),
            "stucks": int(stuck_history[0]) + sum(not a and b for a, b in zip(stuck_history, stuck_history[1:])),
        }
        for key, expected in counts.items():
            if type(metrics[key]) is not int or metrics[key] != expected:
                raise ValueError(f"Metrik {key} SA tidak konsisten")
        frequency = counts["stuck_i"] / iterations if iterations else 0.0
        if not _same(metrics["stuck_frequency"], frequency):
            raise ValueError("Frekuensi stuck SA tidak konsisten")

        temperature = temperature_schedule(iterations + 1, config)
        if not _same(metrics["final_temperature"], temperature, abs_tol=0):
            raise ValueError("Temperatur akhir SA tidak sesuai schedule")
        if temperature == 0:
            stopr = "iteration_limit" if iterations == config["iterations"] else "temperature_limit"
        else:
            stopr = "sampling_limit" if problem.packages else "empty_problem"
        if result.stop_reason != stopr:
            raise ValueError("Alasan berhenti SA tidak sesuai")
        failures = int(stopr in ("sampling_limit", "empty_problem"))
        if type(metrics["failures_sample"]) is not int or metrics["failures_sample"] != failures:
            raise ValueError("Jumlah sampling gagal SA tidak sesuai")
        attempts = metrics["attempts_sample"]
        failed_attempts = config["max_attempts"] if stopr == "sampling_limit" else 0
        if (type(attempts) is not int or not iterations + failed_attempts <= attempts <= iterations * config["max_attempts"] + failed_attempts):
            raise ValueError("Jumlah percobaan sampling SA tidak sesuai")
    else:
        size, generations = config["population_size"], config["generations"]
        means = metrics["mean_history"]
        if len(history) != generations + 1 or len(means) != len(history):
            raise ValueError("Histori GA harus G+1")
        if list(history) != metrics["max_history"] or any(not math.isfinite(a) or a < 0 or a > b + 1e-10 for a, b in zip(means, history)):
            raise ValueError("Mean/max GA tidak konsisten")
        if metrics["population_sizes"] != [size] * len(history):
            raise ValueError("Ukuran populasi GA berubah")
        for label, index in (("initial", 0), ("final", -1)):
            population = metrics[label + "_population"]
            if len(population) != size:
                raise ValueError("Ukuran snapshot populasi tidak sesuai")
            scores = [objective(problem, State.from_dict(problem, item), config["objective"]) for item in population]
            if len(metrics[label + "_population_scores"]) != size or not all(_same(a, b) for a, b in zip(scores, metrics[label + "_population_scores"])):
                raise ValueError("Skor populasi tidak sesuai")
            if not _same(max(scores), history[index]) or not _same(sum(scores) / size, means[index]):
                raise ValueError("Statistik populasi tidak sesuai")
        if not _same(history[-1], result.final_score):
            raise ValueError("Skor final GA tidak sesuai populasi terakhir")


# ubah hasil run jadi dict siap disimpan ke JSON (plus info laptop & peta sel truk)
def to_document(problem: Problem, result: RunResult, case_name: str):
    check_result(problem, result)
    run = {name: getattr(result, name) for name in (
        "algorithm", "seed", "initial_score", "final_score", "best_score", "duration_seconds", "stop_reason")}
    run.update(config=dict(result.config), metrics=dict(result.metrics),
               objective_history=list(result.objective_history))
    for name in ("initial_state", "final_state", "best_state"):
        state = getattr(result, name)
        run[name] = state.to_dict() if state is not None else None
    return {
        "schema_version": 1, "case_name": case_name, "problem_sha256": problem_hash(problem),
        "problem": problem.to_dict(), "run": run,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "machine": platform.machine(), "timing": "perf_counter; initialization and search, excluding I/O/plots"},
        "occupancy": {label: [[*cell, package_id] for cell, package_id in occupancy(problem, state).items()]
                      for label, state in (("initial", result.initial_state), ("final", result.final_state))},
    }


# kebalikan to_document: dict JSON dibalikin jadi objek, sekalian dicek nggak ada yang diutak-atik
def from_document(document):
    if document.get("schema_version") != 1:
        raise ValueError("Versi schema hasil tidak didukung")
    problem = Problem.from_dict(document["problem"])
    if document["problem_sha256"] != problem_hash(problem):
        raise ValueError("Hash input tidak cocok")
    data = dict(document["run"])
    for name in ("initial_state", "final_state", "best_state"):
        if data[name] is not None:
            data[name] = State.from_dict(problem, data[name])
    data["objective_history"] = tuple(data["objective_history"])
    result = RunResult(**data)
    check_result(problem, result)
    for label, state in (("initial", result.initial_state), ("final", result.final_state)):
        expected = [[*cell, package_id] for cell, package_id in occupancy(problem, state).items()]
        if document["occupancy"][label] != expected:
            raise ValueError(f"Occupancy {label} tidak sesuai state")
    return problem, result


# buka result.json dari disk, balikin (problem, hasil run, isi JSON mentah)
def load_result(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    problem, result = from_document(document)
    return problem, result, document


# simpan hasil run ke file result.json
def save_result(path, problem, result, case_name):
    document = to_document(problem, result, case_name)
    Path(path).write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
