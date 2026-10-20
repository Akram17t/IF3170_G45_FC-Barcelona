import math
import sys
from random import Random
from time import perf_counter

from .common import configuration, initial_state
from ..model import RunResult
from ..neighbors import improving_neighbor_exists, sample_feasible_neighbor
from ..validation import objective

def acceptance_probability(delta, temperature): #peluang diterima yang e^(delta/t)
        if(isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta)):
                raise ValueError("delta harus angka beneran (bukan infinite)")

        if(isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0):
                raise ValueError("temperature harus angka beneran (bukan infinite dan positif)")

        if delta > 0:
                return 1.0

        return math.exp(delta / temperature)

def temperature_schedule(t, config): #milih temperature based on iterasi, pakai tempAwal * rate^(iterasi-1). justifikasinya karena dengan ini temp turun bertahap dan proporsional
        if t > config["iterations"]:
                return 0.0

        temperature = (config["initial_temperature"] * config["cooling_rate"] ** (t-1))

        if temperature <= config["min_temperature"]:
                return 0.0

        return temperature

def _exponential_metrics(delta, temperature): #buat grafik purposes
        log_ratio = delta / temperature
        log_overflow = not math.isfinite(log_ratio)
        exp_overflow = log_ratio > math.log(sys.float_info.max)

        ratio = None if exp_overflow else math.exp(log_ratio)

        return {"exp_delta_over_t": ratio,
                "log_acceptance_ratio": None if log_overflow else log_ratio,
                "log_ratio_overflow": log_overflow,
                "exp_overflow": exp_overflow,}

def simulated_annealing(problem, config=None, initial=None):
        started = perf_counter()
        if config is None:
                config = configuration("sa", {})
        else:
                config = configuration("sa", config)
        randomm = Random(config["seed"]) #ini random number generator

        first = initial_state(problem, randomm, initial)
        current = first
        score = objective(problem, current, config["objective"])
        best = current #buat catet dulu aja
        best_score = score
        history = [score]
        records = []
        accepted_moves = 0
        accepted_moves_worse = 0 #jelek tp probsnya masi masuk
        attempts_sample = 0
        failures_sample = 0 
        # Stuck di sini juga plateau and global optimum
        stuck = not improving_neighbor_exists(problem, current, config["objective"])
        stuck_history = [stuck]
        stucks = int(stuck)
        stuck_i = 0
        t = 1

        while True:
                temperature = temperature_schedule(t, config)
                if temperature == 0:
                        if t > config["iterations"]:
                                stopr = "iteration_limit"
                        else:
                                stopr = "temperature_limit"
                        break
                samples = sample_feasible_neighbor(problem, current, randomm, config["max_attempts"]) #successor acak
                attempts_sample += samples.attempts

                if samples.state is None:
                        failures_sample += 1
                        if samples.reason == "empty_problem":
                                stopr = "empty_problem"
                        else:
                                stopr = "sampling_limit"
                        break

                next_state = samples.state
                next_score = objective(problem, next_state, config["objective"])
                delta = next_score - score

                if delta > 0:
                        accepted = True
                        probability = 1.0 #kandidat lebih baik, langsung diterima
                else:
                        probability = acceptance_probability(delta, temperature)
                        accepted = randomm.random() < probability
                
                previous_stuck = stuck

                if accepted:
                        current = next_state
                        score = next_score
                        accepted_moves += 1

                        if delta < 0:
                                accepted_moves_worse += 1

                        if score > best_score:
                                best = current
                                best_score = score
                        stuck = not improving_neighbor_exists(problem, current, config["objective"])

                if stuck:
                        stuck_i += 1

                        if not previous_stuck:
                                stucks += 1

                history.append(score)
                stuck_history.append(stuck)
                records.append({"iteration": t,
                                "temperature": temperature,
                                "delta": delta,
                                "accepted": accepted,
                                "objective": score,
                                "acceptance_probability": probability,
                                "stuck": stuck,
                                **_exponential_metrics(delta, temperature),})
                t += 1
        iterations = len(records)
        return RunResult(algorithm="sa",
                         seed=config["seed"],
                         config=config,
                         initial_state=first,
                         final_state=current,
                         best_state=best,
                         initial_score=history[0],
                         final_score=score,
                         best_score=best_score,
                         objective_history=tuple(history),
                         duration_seconds=perf_counter()-started,
                         stop_reason=stopr,
                         metrics={"iterations": iterations,
                                  "records": records,
                                  "final_current": current.to_dict(),
                                  "final_current_score": score,
                                  "final_temperature": temperature,
                                  "accepted_moves": accepted_moves,
                                  "accepted_moves_worse": accepted_moves_worse,
                                  "rejected_moves": iterations-accepted_moves,
                                  "attempts_sample": attempts_sample,
                                  "failures_sample": failures_sample,
                                  "initial_stuck": stuck_history[0],
                                  "stuck_history": stuck_history,
                                  "stuck_i": stuck_i,
                                  "stucks": stucks,
                                  "stuck_frequency": (stuck_i/iterations if iterations else 0.0), "stuck_definition": ("Tidak ada neighbor valid dengan objective lebih tinggi. Mencakup plateau dan global optimum. Initial state dapat memulai episode, tetapi tidak dihitung sebagai iterasi"),}
                        )


def main():
        import argparse
        from pathlib import Path
        from ..model import load_problem
        from ..results import sa_directory, save_result
        from ..visualization import render_run

        root = Path(__file__).resolve().parents[2]
        defaults = configuration("sa", {})
        parser = argparse.ArgumentParser(description="Simulated Annealing untuk penataan paket dalam truk")
        parser.add_argument("--input", type=Path, default=root / "examples" / "tiny.json")
        parser.add_argument("--output", type=Path, default=root / "output" / "sa")
        parser.add_argument("--seed", type=int, default=defaults["seed"])
        parser.add_argument("--iterations", type=int, default=defaults["iterations"])
        parser.add_argument("--max-attempts", type=int, default=defaults["max_attempts"])
        parser.add_argument("--initial-temperature", type=float, default=defaults["initial_temperature"])
        parser.add_argument("--cooling-rate", type=float, default=defaults["cooling_rate"])
        parser.add_argument("--min-temperature", type=float, default=defaults["min_temperature"])
        args = parser.parse_args()

        problem = load_problem(args.input)
        options = dict(seed=args.seed, iterations=args.iterations, max_attempts=args.max_attempts,
                       initial_temperature=args.initial_temperature, cooling_rate=args.cooling_rate,
                       min_temperature=args.min_temperature)
        result = simulated_annealing(problem, options)

        #Simpen hasil dan bikin visualisasi setelah pencarian, biar ga masuk durasi run
        directory = sa_directory(args.output, problem, args.input.stem, args.seed)
        save_result(directory / "result.json", problem, result, args.input.stem)
        render_run(problem, result, directory, args.input.stem)

        metrics = result.metrics
        print(f"SA: {result.initial_score:g} -> {result.final_score:g}")
        print(f"Best yang pernah ditemukan: {result.best_score:g}")
        print(f"Iterasi: {metrics['iterations']} | Stop: {result.stop_reason}")
        print(f"Move memburuk diterima: {metrics['accepted_moves_worse']}")
        print(f"Stuck: {metrics['stucks']} episode | {metrics['stuck_i']} iterasi ({metrics['stuck_frequency']:.1%})")
        print(f"Durasi: {result.duration_seconds:.4f}s | Output: {directory}")


if __name__ == "__main__":
        main()

