from random import Random
import unittest
from unittest.mock import patch

from src.algorithms import ga, hc, sa
from src.algorithms.common import configuration
from src.model import Problem, State, Truck
from src.neighbors import NeighborResult
from src.results import check_result
from src.validation import validate
from tests.test_core import package, state


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.problem = Problem(Truck(2, 1, 1, 2), (package("A", value=5), package("B", value=4)))

    def test_hc_accepts_only_improvement_and_counts_iterations(self):
        empty = State.empty(self.problem)
        one = state(self.problem, {"A": (0, 0, 0)})
        both = state(self.problem, {"A": (0, 0, 0), "B": (1, 0, 0)})
        neighbors = [NeighborResult(s, None, 1, 0, "found") for s in (one, empty, both)]
        with patch("src.algorithms.hc.sample_feasible_neighbor", side_effect=neighbors):
            result = hc.run(self.problem, {"iterations": 3}, Random(0), empty)
        self.assertEqual(result.objective_history, (0, 5, 5, 9))
        self.assertEqual(result.metrics["accepted"], 2)
        self.assertEqual(result.metrics["rejected_non_improving"], 1)
        self.assertEqual(empty, State.empty(self.problem))
        check_result(self.problem, result)

    def test_sampling_failure_is_not_claimed_as_local_optimum(self):
        failure = NeighborResult(None, None, 2, 2, "attempt_limit")
        with patch("src.algorithms.hc.sample_feasible_neighbor", return_value=failure):
            result = hc.run(self.problem, {"iterations": 3}, Random(0), State.empty(self.problem))
        self.assertEqual(result.stop_reason, "attempt_limit")
        self.assertEqual(result.metrics["iterations"], 0)

    def test_sa_probability_and_best_survives_downhill(self):
        self.assertEqual(sa.acceptance_probability(5, 1), 1)
        self.assertEqual(sa.acceptance_probability(0, 1), 1)
        self.assertGreater(sa.acceptance_probability(-5, 100), sa.acceptance_probability(-5, 1))
        empty = State.empty(self.problem)
        one = state(self.problem, {"A": (0, 0, 0)})
        neighbors = [NeighborResult(s, None, 1, 0, "found") for s in (one, empty)]
        rng = Random(0)
        with patch.object(rng, "random", return_value=0), patch("src.algorithms.sa.sample_feasible_neighbor", side_effect=neighbors):
            result = sa.run(self.problem, {"iterations": 2}, rng, empty)
        self.assertEqual(result.objective_history, (0, 5, 0))
        self.assertEqual(result.final_score, 5)
        self.assertEqual(result.metrics["final_current_score"], 0)
        self.assertGreater(result.metrics["records"][0]["exp_delta_over_t"], 1)
        check_result(self.problem, result)

    def test_sa_temperature_termination_and_overflow_record(self):
        empty = State.empty(self.problem)
        one = state(self.problem, {"A": (0, 0, 0)})
        neighbor = NeighborResult(one, None, 1, 0, "found")
        with patch("src.algorithms.sa.sample_feasible_neighbor", return_value=neighbor):
            result = sa.run(self.problem, {"iterations": 1, "initial_temperature": 1e-320,
                                           "min_temperature": 1e-320}, Random(0), empty)
        self.assertTrue(result.metrics["records"][0]["exp_overflow"])
        self.assertIsNone(result.metrics["records"][0]["exp_delta_over_t"])
        check_result(self.problem, result)
        result = sa.run(self.problem, {"iterations": 10, "initial_temperature": 1,
                                       "min_temperature": .75, "cooling_rate": .5}, Random(0))
        self.assertEqual(result.stop_reason, "temperature_limit")
        self.assertEqual(result.metrics["iterations"], 1)

    def test_ga_repair_overlap_overweight_fragile_and_parent_unchanged(self):
        for problem, raw in (
            (self.problem, {"A": (0, 0, 0), "B": (0, 0, 0)}),
            (Problem(Truck(2, 1, 1, 1), self.problem.packages), {"A": (0, 0, 0), "B": (1, 0, 0)}),
            (Problem(Truck(1, 1, 2, 2), (package("A", fragile=True), package("B"))), {"A": (0, 0, 0), "B": (0, 0, 1)}),
        ):
            child = state(problem, raw)
            snapshot = child.to_dict()
            repaired, dropped = ga.repair(problem, child, Random(0))
            self.assertTrue(validate(problem, repaired).valid)
            self.assertGreater(dropped, 0)
            self.assertEqual(set(repaired.placements), {"A", "B"})
            self.assertEqual(child.to_dict(), snapshot)
        first, second = State.empty(self.problem), state(self.problem, {"A": (0, 0, 0)})
        a, b = first.to_dict(), second.to_dict()
        children = ga.crossover(self.problem, first, second, Random(1))
        self.assertEqual(first.to_dict(), a)
        self.assertEqual(second.to_dict(), b)
        # Satu titik: anak pertama A dari first, B dari second; anak kedua kebalikannya.
        self.assertEqual([c.placements["A"] for c in children], [first.placements["A"], second.placements["A"]])
        self.assertEqual([c.placements["B"] for c in children], [second.placements["B"], first.placements["B"]])

    def test_roulette_matches_class_example(self):
        class Fixed(Random):
            def __init__(self, value):
                super().__init__(0)
                self.value = value

            def random(self):
                return self.value

        population = ["P1", "P2", "P3", "P4"]
        scores = [19, 14, 19, 0]  # total 52: P1 0-36.5%, P2 36.5-63.5%, P3 63.5-100%, P4 0%
        self.assertEqual(ga.roulette(population, scores, Fixed(0.20)), "P1")
        self.assertEqual(ga.roulette(population, scores, Fixed(0.75)), "P3")
        self.assertEqual(ga.roulette(population, scores, Fixed(0.999)), "P3")
        self.assertIn(ga.roulette(population, [0, 0, 0, 0], Random(0)), population)

    def test_ga_population_invariants_and_zero_fitness_selection(self):
        result = ga.run(self.problem, {"population_size": 4, "generations": 5}, Random(0))
        self.assertEqual(result.metrics["population_sizes"], [4] * 6)
        self.assertEqual(result.metrics["objective_evaluations"], 24)
        self.assertTrue(all(a <= b for a, b in zip(result.objective_history, result.objective_history[1:])))
        self.assertTrue(all(a <= b for a, b in zip(result.metrics["mean_history"], result.objective_history)))
        check_result(self.problem, result)
        zero = Problem(Truck(1, 1, 1, 0), (package("Z", value=0),))
        result = ga.run(zero, {"population_size": 2, "generations": 2}, Random(0))
        self.assertEqual(result.final_score, 0)

    def test_all_algorithms_seed_reproducibility_and_empty_problem(self):
        for name, algorithm in (("hc", hc), ("sa", sa), ("ga", ga)):
            config = {"seed": 7, **({"generations": 3, "population_size": 4} if name == "ga" else {"iterations": 12})}
            first = algorithm.run(self.problem, config, Random(7))
            second = algorithm.run(self.problem, config, Random(7))
            self.assertEqual(first.final_state, second.final_state)
            self.assertEqual(first.objective_history, second.objective_history)
            self.assertEqual(first.metrics, second.metrics)
            check_result(self.problem, first)
            empty = Problem(Truck(1, 1, 1, 0), ())
            result = algorithm.run(empty, config, Random(7))
            self.assertEqual(result.final_score, 0)
            check_result(empty, result)

    def test_invalid_configs(self):
        for name, config in (("hc", {"iterations": -1}), ("hc", {"iterations": True}),
                             ("sa", {"initial_temperature": 0}), ("sa", {"cooling_rate": 1.1}),
                             ("sa", {"initial_temperature": float("nan")}),
                             ("ga", {"population_size": 1}), ("ga", {"mutation_rate": 2}),
                             ("ga", {"elitism": 20}), ("hc", {"unknown": 1})):
            with self.subTest(name=name, config=config), self.assertRaises(ValueError):
                configuration(name, config)


if __name__ == "__main__":
    unittest.main()
