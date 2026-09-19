"""Tool receipt writer: timestamped JSON under receipts/tools/<tool>/."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from .schemas import ToolReceipt, build_tool_receipt, validate_tool_receipt


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_tool_name(tool: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(tool).strip())
    name = name.strip("._-") or "tool"
    return name


def _timestamp_slug() -> str:
    # Compact UTC stamp safe for filenames.
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def write_tool_receipt(
    tool: str,
    payload: Union[Mapping[str, Any], ToolReceipt],
    *,
    root: Optional[Path] = None,
    status: str = "ok",
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write receipts/tools/<tool>/<timestamp>.json and update latest.json."""
    base = Path(root) if root is not None else _project_root()
    tool_name = _safe_tool_name(tool)
    out_dir = base / "receipts" / "tools" / tool_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(payload, ToolReceipt):
        receipt = payload
        if receipt.tool != tool_name and receipt.tool != tool:
            # Prefer caller tool name for path consistency.
            data = receipt.to_dict()
            data["tool"] = tool_name
            receipt = ToolReceipt.from_dict(data)
    else:
        # Allow full receipt dicts or bare payload dicts.
        if isinstance(payload, Mapping) and "tool" in payload and "payload" in payload:
            data = dict(payload)
            data["tool"] = tool_name
            if status and "status" not in data:
                data["status"] = status
            receipt = ToolReceipt.from_dict(validate_tool_receipt(data))
        else:
            kwargs = dict(extra or {})
            receipt = build_tool_receipt(tool_name, dict(payload), status=status, **kwargs)

    body_obj = receipt.to_dict()
    body = json.dumps(body_obj, indent=2, sort_keys=True, ensure_ascii=False) + chr(10)

    ts = _timestamp_slug()
    stamped = out_dir / f"{ts}.json"
    # Avoid clobbering if called twice in the same second.
    if stamped.exists():
        n = 1
        while True:
            alt = out_dir / f"{ts}_{n:02d}.json"
            if not alt.exists():
                stamped = alt
                break
            n += 1

    stamped.write_text(body, encoding="utf-8")
    (out_dir / "latest.json").write_text(body, encoding="utf-8")
    return stamped
