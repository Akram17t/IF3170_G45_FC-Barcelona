from itertools import combinations
import unittest

from src.analysis import _knapsack_bound, value_upper_bound
from src.model import Problem, Truck
from tests.test_core import package


class BoundTests(unittest.TestCase):
    def test_integer_bound_matches_subset_enumeration(self):
        values, costs, capacity = [4, 8, 3, 10], [0, 2, 1, 3], 3
        exhaustive = max(sum(values[i] for i in subset)
                         for n in range(5) for subset in combinations(range(4), n)
                         if sum(costs[i] for i in subset) <= capacity)
        self.assertEqual(_knapsack_bound(values, costs, capacity), exhaustive)

    def test_fractional_bound_and_zero_capacity(self):
        self.assertAlmostEqual(_knapsack_bound([6, 5], [.6, .7], 1), 6 + 5 * .4 / .7)
        p = Problem(Truck(1, 1, 1, 0), (package("A", value=8, weight=0), package("B", value=9, weight=0)))
        self.assertEqual(value_upper_bound(p)["value"], 9)


if __name__ == "__main__":
    unittest.main()
