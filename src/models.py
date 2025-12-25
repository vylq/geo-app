from __future__ import annotations

from dataclasses import dataclass, asdict
from uuid import uuid4


@dataclass
class Place:
    id: str
    name: str
    latitude: float
    longitude: float

    @staticmethod
    def create(name: str, latitude: float, longitude: float) -> "Place":
        return Place(
            id=str(uuid4()),
            name=name.strip(),
            latitude=float(latitude),
            longitude=float(longitude),
        )

    def to_dict(self) -> dict:
        return asdict(self)
