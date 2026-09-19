"""AxonForge light switch — manual on/off + autonomous demand mode."""
from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "receipts" / "axonforge"
STATE_PATH = DATA / "switch.json"
TOOLS = Path(__file__).resolve().parents[1] / "tools"


@dataclass
class SwitchState:
    powered_on: bool = True
    autonomous: bool = False
    last_flip_s: float = 0.0
    last_reason: str = "default_on"
    flips: int = 0


class LightSwitch:
    """Power gate for AxonForge.

    Manual: operator flips on/off.
    Autonomous: ON when work is pending/running; OFF after idle grace.
    """

    IDLE_GRACE_S = 8.0
    POLL_S = 1.5

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state = self._load()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._idle_since: float | None = None
        self._work_probe = lambda: (False, False)
        if self.state.autonomous:
            self._ensure_loop()

    def _load(self) -> SwitchState:
        DATA.mkdir(parents=True, exist_ok=True)
        if STATE_PATH.exists():
            raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            return SwitchState(
                powered_on=bool(raw.get("powered_on", True)),
                autonomous=bool(raw.get("autonomous", False)),
                last_flip_s=float(raw.get("last_flip_s", 0.0)),
                last_reason=str(raw.get("last_reason", "")),
                flips=int(raw.get("flips", 0)),
            )
        return SwitchState()

    def _persist_unlocked(self) -> None:
        DATA.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(asdict(self.state), indent=2) + "\n", encoding="utf-8")
        try:
            if str(TOOLS) not in sys.path:
                sys.path.insert(0, str(TOOLS))
            import registry as tools_registry
            tools_registry.set_power(
                "axonforge",
                powered_on=self.state.powered_on,
                autonomous=self.state.autonomous,
            )
        except Exception:  # noqa: BLE001
            pass

    def bind_work_probe(self, probe) -> None:
        self._work_probe = probe

    def as_dict(self) -> dict:
        with self._lock:
            d = asdict(self.state)
            d["mode"] = "autonomous" if self.state.autonomous else "manual"
            return d

    def is_on(self) -> bool:
        with self._lock:
            return bool(self.state.powered_on)

    def set_power(self, on: bool, *, reason: str = "manual") -> dict:
        with self._lock:
            if self.state.powered_on != on:
                self.state.powered_on = on
                self.state.last_flip_s = time.time()
                self.state.flips += 1
            self.state.last_reason = reason
            self._persist_unlocked()
            d = asdict(self.state)
            d["mode"] = "autonomous" if self.state.autonomous else "manual"
            return d

    def toggle(self, *, reason: str = "manual_toggle") -> dict:
        with self._lock:
            on = not self.state.powered_on
        return self.set_power(on, reason=reason)

    def set_autonomous(self, enabled: bool) -> dict:
        with self._lock:
            self.state.autonomous = enabled
            self.state.last_reason = "autonomous_enabled" if enabled else "autonomous_disabled"
            self.state.last_flip_s = time.time()
            self._persist_unlocked()
            d = asdict(self.state)
            d["mode"] = "autonomous" if self.state.autonomous else "manual"
        if enabled:
            self._ensure_loop()
        else:
            self._stop.set()
        return d

    def _ensure_loop(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._autonomous_loop, name="af-light-switch", daemon=True)
        self._thread.start()

    def _autonomous_loop(self) -> None:
        while not self._stop.is_set():
            try:
                with self._lock:
                    auto = self.state.autonomous
                if not auto:
                    break
                has_pending, is_running = self._work_probe()
                busy = bool(has_pending or is_running)
                if busy:
                    self._idle_since = None
                    if not self.is_on():
                        self.set_power(True, reason="autonomous_demand")
                else:
                    now = time.time()
                    if self._idle_since is None:
                        self._idle_since = now
                    elif (now - self._idle_since) >= self.IDLE_GRACE_S and self.is_on():
                        self.set_power(False, reason="autonomous_idle")
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.POLL_S)


SWITCH = LightSwitch()
