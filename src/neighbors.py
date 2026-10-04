# invalid or no op direturn as None
from dataclasses import dataclass
from enum import Enum
from itertools import product
from random import Random

from .initialize import candidate_position
from .model import Placement, Position, Problem, State, oriented_size, rotate
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


def apply_move(problem: Problem, state: State, move: Move) -> State | None:
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
