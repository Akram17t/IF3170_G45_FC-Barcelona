# Move yang ga valid atau ga mengubah state return none
from dataclasses import dataclass
from enum import Enum

from .initialize import candidate_position
from .model import Placement, Position, State, oriented_size, rotate
from .validation import require_valid, validate


class MoveKind(str, Enum):
    SWAP = "swap"
    RELOCATE = "relocate"
    ROTATE = "rotate"


@dataclass(frozen=True)
class Move:
    kind: MoveKind
    package_id: str
    other_id: str | None = None
    position: Position | None = None
    axis: str | None = None


@dataclass(frozen=True)
class NeighborResult:
    state: State | None
    move: Move | None
    attempts: int
    rejected: int
    reason: str


def apply_move(problem, state, move):
    # Move di copy state biar yang awal ga berubah
    require_valid(problem, state)
    if move.package_id not in state.placements:
        raise ValueError("ID move tidak dikenal")
    placements = dict(state.placements)
    first = placements[move.package_id]
    if move.kind == MoveKind.SWAP:
        if move.other_id not in placements:
            raise ValueError("ID pasangan swap tidak dikenal")
        second = placements[move.other_id]
        # Anchor dituker, orientasi ngikutin masing2 package
        placements[move.package_id] = Placement(second.position, first.orientation)
        placements[move.other_id] = Placement(first.position, second.orientation)
    elif move.kind == MoveKind.RELOCATE:
        placements[move.package_id] = Placement(move.position, first.orientation)
    elif move.kind == MoveKind.ROTATE:
        placements[move.package_id] = Placement(first.position, rotate(first.orientation, move.axis))
    else:
        raise ValueError("Jenis move tidak dikenal")
    candidate = State(placements)
    if candidate == state or not validate(problem, candidate).valid:
        return None
    return candidate


def sample_feasible_neighbor(problem, state, rng, max_attempts=100):
    # Coba move random sampe dapet candidate valid
    require_valid(problem, state)
    if (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or max_attempts <= 0
    ):
        raise ValueError("max_attempts harus integer positif")
    packages = list(problem.packages)
    if not packages:
        return NeighborResult(None, None, 0, 0, "empty_problem")
    for attempt in range(1, max_attempts + 1):
        kind = rng.choice(list(MoveKind))
        package = rng.choice(packages)
        if kind == MoveKind.SWAP:
            if len(packages) < 2:
                continue
            other_packages = []
            for other_package in packages:
                if other_package.id != package.id:
                    other_packages.append(other_package)
            other = rng.choice(other_packages)
            move = Move(kind, package.id, other_id=other.id)
        elif kind == MoveKind.ROTATE:
            axis = rng.choice("xyz")
            move = Move(kind, package.id, axis=axis)
        else:
            current = state.placements[package.id]
            position = candidate_position(problem, state, package, current.orientation, rng)
            # Paket di dalam truck cuma 20% buat dikeluarin
            if current.position is not None and rng.random() < 0.2:
                position = None
            move = Move(kind, package.id, position=position)
        candidate = apply_move(problem, state, move)
        if candidate is not None:
            return NeighborResult(candidate, move, attempt, attempt - 1, "found")
    return NeighborResult(None, None, max_attempts, max_attempts, "attempt_limit")


def improving_neighbor_exists(problem, state):
    # Cek cuma insert dan swap masuk/luar bisa naikin value

    require_valid(problem, state)
    inside = []
    outside = []
    for package in problem.packages:
        if state.placements[package.id].position is None:
            outside.append(package)
        else:
            inside.append(package)

    for package in outside:
        # Coba paket + di setiap posisi int.
        if package.value > 0:
            orientation = state.placements[package.id].orientation
            width, length, height = oriented_size(package, orientation)
            truck_width, truck_length, truck_height = problem.truck.dimensions

            for x in range(truck_width - width + 1):
                for y in range(truck_length - length + 1):
                    for z in range(truck_height - height + 1):
                        position = (x, y, z)
                        move = Move(MoveKind.RELOCATE, package.id, position=position)
                        candidate = apply_move(problem, state, move)
                        if candidate is not None:
                            return True

        # Coba swap paket yang nilainya lebih kecil
        for other in inside:
            if package.value > other.value:
                move = Move(MoveKind.SWAP, package.id, other_id=other.id)
                candidate = apply_move(problem, state, move)
                if candidate is not None:
                    return True

    return False
