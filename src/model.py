"""Data bersama untuk masalah penataan paket.

Sumbu truk adalah (x, y, z) = (width, length, height). Orientasi adalah
pemetaan sisi asli paket (w, l, h) ke sumbu truk. Aturan kelayakan geometris
dan objective berada di src.validation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


Number = int | float
Position = tuple[int, int, int]


class Orientation(str, Enum):
    """Urutan huruf menunjukkan sumbu tujuan untuk sisi asli w, l, h."""

    XYZ = "xyz"
    XZY = "xzy"
    YXZ = "yxz"
    YZX = "yzx"
    ZXY = "zxy"
    ZYX = "zyx"


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} harus berupa object")
    return value


def _fields(data: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = expected - data.keys()
    extra = data.keys() - expected
    if missing or extra:
        raise ValueError(
            f"{label}: field kurang {sorted(missing)}, field tidak dikenal {sorted(extra)}"
        )


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} harus bilangan bulat positif")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} harus bilangan bulat tidak negatif")
    return value


def _nonnegative_number(value: object, label: str) -> Number:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (isinstance(value, float) and not math.isfinite(value))
        or value < 0
    ):
        raise ValueError(f"{label} harus bilangan hingga yang tidak negatif")
    return value


def _orientation(value: object) -> Orientation:
    if isinstance(value, Mapping):
        _fields(value, {"w", "l", "h"}, "orientation")
        if any(value[side] not in ("x", "y", "z") for side in ("w", "l", "h")):
            raise ValueError("orientation harus memetakan setiap sisi ke satu sumbu x/y/z")
        value = "".join(value[side] for side in ("w", "l", "h"))
    try:
        return Orientation(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("orientation harus salah satu dari enam permutasi xyz") from exc


@dataclass(frozen=True)
class Truck:
    w: int
    l: int
    h: int
    max_capacity: Number

    def __post_init__(self) -> None:
        for name in ("w", "l", "h"):
            _positive_int(getattr(self, name), f"truck.{name}")
        _nonnegative_number(self.max_capacity, "truck.max_capacity")

    @property
    def dimensions(self) -> tuple[int, int, int]:
        return self.w, self.l, self.h

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Truck:
        data = _object(data, "truck")
        _fields(data, {"dimensions", "maxCapacity"}, "truck")
        dimensions = _object(data["dimensions"], "truck.dimensions")
        _fields(dimensions, {"w", "l", "h"}, "truck.dimensions")
        return cls(
            w=dimensions["w"],
            l=dimensions["l"],
            h=dimensions["h"],
            max_capacity=data["maxCapacity"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimensions": {"w": self.w, "l": self.l, "h": self.h},
            "maxCapacity": self.max_capacity,
        }


@dataclass(frozen=True)
class Package:
    id: str
    w: int
    l: int
    h: int
    value: Number
    weight: Number
    is_fragile: bool
    eta: int
    orientation: Orientation

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("package.id harus string tidak kosong")
        for name in ("w", "l", "h"):
            _positive_int(getattr(self, name), f"package {self.id}.{name}")
        _nonnegative_number(self.value, f"package {self.id}.value")
        _nonnegative_number(self.weight, f"package {self.id}.weight")
        if not isinstance(self.is_fragile, bool):
            raise ValueError(f"package {self.id}.is_fragile harus boolean")
        _nonnegative_int(self.eta, f"package {self.id}.eta")
        if not isinstance(self.orientation, Orientation):
            raise ValueError(f"package {self.id}.orientation tidak sah")

    @property
    def dimensions(self) -> tuple[int, int, int]:
        return self.w, self.l, self.h

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Package:
        data = _object(data, "package")
        _fields(
            data,
            {"id", "dimensions", "orientation", "value", "weight", "isFragile", "ETA"},
            "package",
        )
        dimensions = _object(data["dimensions"], "package.dimensions")
        _fields(dimensions, {"w", "l", "h"}, "package.dimensions")
        return cls(
            id=data["id"],
            w=dimensions["w"],
            l=dimensions["l"],
            h=dimensions["h"],
            value=data["value"],
            weight=data["weight"],
            is_fragile=data["isFragile"],
            eta=data["ETA"],
            orientation=_orientation(data["orientation"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dimensions": {"w": self.w, "l": self.l, "h": self.h},
            "orientation": self.orientation.value,
            "value": self.value,
            "weight": self.weight,
            "isFragile": self.is_fragile,
            "ETA": self.eta,
        }


@dataclass(frozen=True)
class Problem:
    truck: Truck
    packages: tuple[Package, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.truck, Truck):
            raise ValueError("problem.truck harus Truck")
        packages = tuple(self.packages)
        if any(not isinstance(package, Package) for package in packages):
            raise ValueError("problem.packages harus berisi Package")
        ids = [package.id for package in packages]
        if len(ids) != len(set(ids)):
            raise ValueError("ID paket harus unik")
        object.__setattr__(self, "packages", packages)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Problem:
        data = _object(data, "problem")
        _fields(data, {"truck", "packages"}, "problem")
        if not isinstance(data["packages"], list):
            raise ValueError("problem.packages harus berupa list")
        return cls(
            truck=Truck.from_dict(data["truck"]),
            packages=tuple(Package.from_dict(item) for item in data["packages"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "truck": self.truck.to_dict(),
            "packages": [package.to_dict() for package in self.packages],
        }

    def package_map(self) -> dict[str, Package]:
        return {package.id: package for package in self.packages}


@dataclass(frozen=True)
class Placement:
    position: Position | None
    orientation: Orientation

    def __post_init__(self) -> None:
        if self.position is not None and (
            not isinstance(self.position, tuple)
            or len(self.position) != 3
            or any(isinstance(x, bool) or not isinstance(x, int) for x in self.position)
        ):
            raise ValueError("position harus tuple tiga bilangan bulat atau None")
        if not isinstance(self.orientation, Orientation):
            raise ValueError("placement.orientation tidak sah")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Placement:
        data = _object(data, "placement")
        _fields(data, {"position", "orientation"}, "placement")
        position = data["position"]
        if position is not None:
            if not isinstance(position, list):
                raise ValueError("placement.position harus list tiga bilangan bulat atau null")
            position = tuple(position)
        return cls(position=position, orientation=_orientation(data["orientation"]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": list(self.position) if self.position is not None else None,
            "orientation": self.orientation.value,
        }


@dataclass(frozen=True)
class State:
    """Semua paket, termasuk yang di luar (position=None)."""

    placements: Mapping[str, Placement]

    def __post_init__(self) -> None:
        placements = dict(self.placements)
        if any(not isinstance(key, str) or not key for key in placements):
            raise ValueError("state.placements harus memakai ID string tidak kosong")
        if any(not isinstance(item, Placement) for item in placements.values()):
            raise ValueError("state.placements harus berisi Placement")
        object.__setattr__(self, "placements", MappingProxyType(placements))

    @classmethod
    def empty(cls, problem: Problem) -> State:
        return cls(
            {
                package.id: Placement(None, package.orientation)
                for package in problem.packages
            }
        )

    @classmethod
    def from_dict(cls, problem: Problem, data: Mapping[str, Any]) -> State:
        data = _object(data, "state")
        _fields(data, {"placements"}, "state")
        placements = _object(data["placements"], "state.placements")
        expected = set(problem.package_map())
        _fields(placements, expected, "state.placements")
        return cls(
            {package_id: Placement.from_dict(item) for package_id, item in placements.items()}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "placements": {
                package_id: placement.to_dict()
                for package_id, placement in self.placements.items()
            }
        }


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunResult:
    """State, skor, histori, durasi, dan metrik; ekspor/audit oleh src.results."""

    algorithm: str
    seed: int
    config: Mapping[str, Any]
    initial_state: State
    final_state: State
    initial_score: Number
    final_score: Number
    objective_history: tuple[Number, ...]
    duration_seconds: float
    stop_reason: str
    best_state: State | None = None
    best_score: Number | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)


def load_problem(path: str | Path) -> Problem:
    """Baca satu kasus JSON sesuai schema Problem.from_dict."""

    with Path(path).open(encoding="utf-8") as stream:
        return Problem.from_dict(json.load(stream))
