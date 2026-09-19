#!/usr/bin/env python3
"""Register account SSH key and attach to all Affinity instances (no secrets printed)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from cloud.vast_provider import VastProvider, load_repo_dotenv  # noqa: E402

load_repo_dotenv(REPO)
pub_path = Path.home() / ".ssh" / "vast_affinity.pub"
if not pub_path.is_file():
    raise SystemExit(f"missing {pub_path}")
pubkey = pub_path.read_text(encoding="utf-8").strip()
# Never print the key — only fingerprint-ish length
print(f"pubkey_len={len(pubkey)} type={pubkey.split()[0] if pubkey else '?'}")

p = VastProvider()
# Account-level register (applies to future instances; may also push to current)
try:
    r = p._request("POST", "ssh/", {"ssh_key": pubkey})
    print("account_ssh:", json.dumps({k: r.get(k) for k in ("success", "msg") if isinstance(r, dict)} or {"raw_type": type(r).__name__}))
except Exception as e:
    print(f"account_ssh_error: {type(e).__name__}: {e}")

instances = p.list_instances()
print(f"n_instances={len(instances)}")
for inst in instances:
    iid = inst.instance_id
    try:
        # v0 attach path
        r = p._request("POST", f"instances/{iid}/ssh/", {"ssh_key": pubkey})
        ok = r.get("success") if isinstance(r, dict) else None
        msg = r.get("msg") if isinstance(r, dict) else str(type(r))
        print(f"attach id={iid} success={ok} msg={msg}")
    except Exception as e:
        # try v1 base
        try:
            r = p._instances_request("POST", f"instances/{iid}/ssh/", {"ssh_key": pubkey})
            ok = r.get("success") if isinstance(r, dict) else None
            msg = r.get("msg") if isinstance(r, dict) else str(type(r))
            print(f"attach_v1 id={iid} success={ok} msg={msg}")
        except Exception as e2:
            print(f"attach_fail id={iid}: {e} | v1: {e2}")
