from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class MethodCfg:
    name: str
    key: str
    path: Path


@dataclass
class SceneCfg:
    scene: str
    target_index: int


@dataclass
class EvaluationCfg:
    methods: List[MethodCfg]
    side_by_side_path: Optional[Path]
    animate_side_by_side: bool
    highlighted: List[SceneCfg]
