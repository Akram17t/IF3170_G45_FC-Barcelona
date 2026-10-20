# Move yang ga valid atau ga mengubah state return none
from dataclasses import dataclass
from enum import Enum
from itertools import combinations

from .initialize import candidate_position
from .model import Placement, Position, State, oriented_size, rotate
from .validation import objective, package_score, require_valid, validate


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


def iter_feasible_neighbors(problem, state):
    # Coba semua move, yang valid dikasih satu2 lewat yield
    require_valid(problem, state)
    # Tiap pasangan dicoba sekali, termasuk paket yang di luar
    for first, second in combinations(problem.packages, 2):
        candidate = apply_move(problem, state, Move(MoveKind.SWAP, first.id, other_id=second.id))
        if candidate is not None:
            yield candidate
    for package in problem.packages:
        placement = state.placements[package.id]
        # Coba rotasi di masing2 sumbu
        for axis in "xyz":
            candidate = apply_move(problem, state, Move(MoveKind.ROTATE, package.id, axis=axis))
            if candidate is not None:
                yield candidate
        # Paket di dalam juga bisa dicoba dikeluarin
        if placement.position is not None:
            candidate = apply_move(problem, state, Move(MoveKind.RELOCATE, package.id))
            if candidate is not None:
                yield candidate
        size = oriented_size(package, placement.orientation)
        limits = tuple(a - b for a, b in zip(problem.truck.dimensions, size))
        if min(limits) < 0:
            continue
        # Coba z=0 atau setinggi atap paket lain biar ga melayang
        levels = {0}
        for other in problem.packages:
            support = state.placements[other.id]
            if other.id != package.id and support.position is not None and not other.is_fragile:
                top = support.position[2] + oriented_size(other, support.orientation)[2]
                if top <= limits[2]:
                    levels.add(top)
        for x in range(limits[0] + 1):
            for y in range(limits[1] + 1):
                for z in sorted(levels):
                    candidate = apply_move(problem, state, Move(
                        MoveKind.RELOCATE, package.id, position=(x, y, z)))
                    if candidate is not None:
                        yield candidate


def best_feasible_neighbor(problem, state, rng, kind="value"):
    # Ambil value tertinggi tanpa nyimpen semua neighbor dalam list
    best, best_score, ties, evaluated = None, float("-inf"), 0, 0
    for candidate in iter_feasible_neighbors(problem, state):
        score = objective(problem, candidate, kind)
        evaluated += 1
        if score > best_score:
            best, best_score, ties = candidate, score, 1
        elif score == best_score:
            # Kalo value sama, pilih random dengan peluang 1/jumlah candidate yang seri
            ties += 1
            if rng.randrange(ties) == 0:
                best = candidate
    return best, best_score, evaluated


def improving_neighbor_exists(problem, state, kind="value"):
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
        if package_score(package, kind) > 0:
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
            if package_score(package, kind) > package_score(other, kind):
                move = Move(MoveKind.SWAP, package.id, other_id=other.id)
                candidate = apply_move(problem, state, move)
                if candidate is not None:
                    return True

    return False
