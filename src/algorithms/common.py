# Configuration and state awal buat semua algoritm nanti

import math
from random import Random

from ..initialize import random_feasible_state
from ..validation import require_valid


DEFAULTS = {
    "hc": {
        "seed": 0,
        "iterations": 500,
        "max_attempts": 100,
    },
    "sa": {
        "seed": 0,
        "iterations": 500,
        "max_attempts": 100,
        "initial_temperature": 20.0,
        "cooling_rate": 0.99,
        "min_temperature": 1e-8,
    },
    "ga": {
        "seed": 0,
        "generations": 40,
        "population_size": 20,
        "mutation_rate": 0.3,
        "elitism": 1,
        "max_attempts": 100,
    },
}


def configuration(algorithm, config):
    # Combine semua parameter pake default dulu, baru nanti validate
    if algorithm not in DEFAULTS:
        raise ValueError("Algoritma harus hc, sa, atau ga")
    unknown = set(config) - set(DEFAULTS[algorithm])
    if unknown:
        raise ValueError(f"Parameter tidak dikenal: {sorted(unknown)}")
    # Copy biar ga ngubah ke DEFAULTS.
    result = DEFAULTS[algorithm].copy()
    result.update(config)
    integer_keys = {
        "seed", "iterations", "generations", "population_size",
        "elitism", "max_attempts",
    }
    zero_allowed_keys = {"iterations", "generations", "elitism"}
    for key, value in result.items():
        if key in integer_keys:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} harus integer")
            # Seed bisa negatif tp parameter lain ada minimum
            if key != "seed":
                minimum = 1
                if key in zero_allowed_keys:
                    minimum = 0
                if value < minimum:
                    raise ValueError(f"{key} di luar batas")
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{key} harus angka hingga")
    if algorithm == "sa":
        if not 0 < result["cooling_rate"] <= 1:
            raise ValueError("cooling_rate harus dalam (0, 1]")
        if not 0 < result["min_temperature"] <= result["initial_temperature"]:
            raise ValueError("Temperatur harus positif, min_temperature <= initial_temperature")
    if algorithm == "ga":
        if result["population_size"] < 2:
            raise ValueError("population_size minimal 2")
        if not 0 <= result["mutation_rate"] <= 1:
            raise ValueError("mutation_rate harus dalam [0, 1]")
        if not 0 <= result["elitism"] < result["population_size"]:
            raise ValueError("elitism harus lebih kecil dari population_size")
    return result


def initial_state(problem, rng, initial):
    # Use state yang dikasih, randomize kalo belum ada
    if not isinstance(rng, Random):
        raise ValueError("rng harus random.Random")
    if initial is not None:
        state = initial
    else:
        state = random_feasible_state(problem, rng)
    require_valid(problem, state)
    return state
