"""Record actual transitions, including unsuccessful moves."""
from dataclasses import dataclass
from typing import Tuple

Position = Tuple[int, int]


@dataclass(frozen=True)
class StepRecord:
    step: int
    from_pos: Position
    action: str
    to_pos: Position


class Memory:
    def __init__(self, initial_pos):
        self.initial_pos = initial_pos
        self.records = []

    def record(self, from_pos, action, to_pos):
        self.records.append(StepRecord(len(self.records) + 1, from_pos, action, to_pos))

    def summary(self):
        trajectory = [self.initial_pos] + [record.to_pos for record in self.records]
        return {"steps": len(self.records), "trajectory": trajectory,
                "actions": [record.action for record in self.records],
                "final_pos": trajectory[-1]}
