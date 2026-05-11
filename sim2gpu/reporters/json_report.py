"""JSON report helpers."""

from __future__ import annotations

import json
from typing import Any


def stable_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"
