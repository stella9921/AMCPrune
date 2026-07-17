from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class AMCImportanceScores:
    """Pre-computed structured importance scores for AMCPrune."""

    unit_scores: torch.Tensor
    unit_ids: List[int]
    unit_names: List[str]
    unit_type: str
    model: str
    score_name: str
    pruning_scope: str
    num_units: int
    metadata: Optional[Dict] = field(default_factory=dict)

    def save(self, path: str) -> None:
        torch.save(asdict(self), path)

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "AMCImportanceScores":
        data = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a dict for importance scores, got {type(data).__name__}")
        data.setdefault("metadata", {})
        return cls(**data)

    @classmethod
    def from_block_rows(
        cls,
        *,
        rows,
        model,
        block_path,
        score_name,
        metadata=None,
    ) -> "AMCImportanceScores":
        ordered = sorted(rows, key=lambda row: row["block"])
        unit_ids = [int(row["block"]) for row in ordered]
        unit_names = [f"{block_path}.{index}" for index in unit_ids]
        unit_scores = torch.tensor(
            [float(row["score"]) for row in ordered],
            dtype=torch.float32,
        )
        return cls(
            unit_scores=unit_scores,
            unit_ids=unit_ids,
            unit_names=unit_names,
            unit_type="block_skip",
            model=model,
            score_name=score_name,
            pruning_scope=block_path,
            num_units=len(unit_ids),
            metadata=metadata or {},
        )

    def to_block_rows(self):
        scores = self.unit_scores.detach().cpu().tolist()
        rows = []
        for unit_id, unit_name, score in zip(self.unit_ids, self.unit_names, scores):
            rows.append({
                "block": int(unit_id),
                "unit_name": unit_name,
                "unit_type": self.unit_type,
                "score": float(score),
            })
        return rows
