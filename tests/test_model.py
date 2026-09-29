import copy
import unittest
from pathlib import Path

from src.model import Orientation, Package, Placement, Problem, State, load_problem


FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "tiny.json"


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = load_problem(FIXTURE)

    def test_tiny_case_and_round_trip(self) -> None:
        self.assertEqual(self.problem.truck.dimensions, (3, 2, 2))
        self.assertEqual([package.id for package in self.problem.packages], ["A", "B", "C"])
        self.assertTrue(self.problem.package_map()["C"].is_fragile)
        self.assertEqual(Problem.from_dict(self.problem.to_dict()), self.problem)

    def test_orientation_mapping_and_package_fields(self) -> None:
        data = self.problem.packages[0].to_dict()
        data["orientation"] = {"w": "y", "l": "x", "h": "z"}
        package = Package.from_dict(data)
        self.assertEqual(package.orientation, Orientation.YXZ)
        self.assertEqual(package.dimensions, (2, 1, 1))
        self.assertEqual(package.eta, 1)

    def test_state_includes_outside_packages_and_is_snapshot(self) -> None:
        state = State.empty(self.problem)
        self.assertEqual(set(state.placements), {"A", "B", "C"})
        self.assertTrue(all(item.position is None for item in state.placements.values()))
        self.assertEqual(State.from_dict(self.problem, state.to_dict()), state)

        source = {"A": Placement((0, 0, 0), Orientation.XYZ)}
        snapshot = State(source)
        source["A"] = Placement(None, Orientation.XYZ)
        self.assertEqual(snapshot.placements["A"].position, (0, 0, 0))
        with self.assertRaises(TypeError):
            snapshot.placements["A"] = Placement(None, Orientation.XYZ)

    def test_duplicate_ids_and_invalid_input_shape(self) -> None:
        data = self.problem.to_dict()
        data["packages"][1]["id"] = "A"
        with self.assertRaisesRegex(ValueError, "ID paket harus unik"):
            Problem.from_dict(data)

        data = self.problem.to_dict()
        data["truck"]["dimensions"]["w"] = True
        with self.assertRaisesRegex(ValueError, "bilangan bulat positif"):
            Problem.from_dict(data)

        data = self.problem.to_dict()
        data["packages"][0]["orientation"] = "xxx"
        with self.assertRaisesRegex(ValueError, "orientation"):
            Problem.from_dict(data)

        data = self.problem.to_dict()
        data["packages"][0]["orientation"] = {"w": "", "l": "xy", "h": "z"}
        with self.assertRaisesRegex(ValueError, "orientation"):
            Problem.from_dict(data)

    def test_state_requires_every_known_id_and_integer_coordinates(self) -> None:
        data = State.empty(self.problem).to_dict()
        del data["placements"]["C"]
        with self.assertRaisesRegex(ValueError, "field kurang"):
            State.from_dict(self.problem, data)

        data = State.empty(self.problem).to_dict()
        data["placements"]["unknown"] = copy.deepcopy(data["placements"]["A"])
        with self.assertRaisesRegex(ValueError, "field tidak dikenal"):
            State.from_dict(self.problem, data)

        data = State.empty(self.problem).to_dict()
        data["placements"]["A"]["position"] = [0, False, 0]
        with self.assertRaisesRegex(ValueError, "position"):
            State.from_dict(self.problem, data)


if __name__ == "__main__":
    unittest.main()
