"""FastAPI integration API for AxonForge."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from axonforge.__version__ import PROJECT, SUBSYSTEM, __version__
from axonforge.runtime import RUNTIME

app = FastAPI(title=f"{PROJECT} — {SUBSYSTEM}", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DASH = Path(__file__).resolve().parents[1] / "axonforge_dashboard"


class SubmitBody(BaseModel):
    shape_zyx: list[int] = Field(default_factory=lambda: [100, 100, 100])
    tile_zyx: list[int] = Field(default_factory=lambda: [32, 64, 64])
    reset: bool = True


class InferBody(BaseModel):
    run_adaptive: bool = True
    use_cascade: bool = False


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "project": PROJECT,
        "subsystem": SUBSYSTEM,
        "version": __version__,
    }


@app.post("/submit_volume")
def submit_volume(body: SubmitBody | None = None) -> dict[str, Any]:
    body = body or SubmitBody()
    return RUNTIME.submit_volume(
        shape_zyx=tuple(body.shape_zyx),  # type: ignore[arg-type]
        tile_zyx=tuple(body.tile_zyx),  # type: ignore[arg-type]
        reset=body.reset,
    )


@app.post("/request_inference")
def request_inference(body: InferBody | None = None) -> dict[str, Any]:
    body = body or InferBody()
    return RUNTIME.request_inference(run_adaptive=body.run_adaptive, use_cascade=body.use_cascade)


@app.get("/status")
def status() -> dict[str, Any]:
    return RUNTIME.status()


@app.get("/demo_run")
def demo_run(
    reset: bool = Query(True),
    return_to: str | None = Query(None),
) -> RedirectResponse:
    """Navigable demo: submit + adaptive run, then bounce back to dashboard."""
    RUNTIME.submit_volume(shape_zyx=(100, 100, 100), tile_zyx=(32, 64, 64), reset=reset)
    RUNTIME.request_inference(run_adaptive=True, use_cascade=False)
    dest = "/?ran=1"
    if return_to:
        # Only allow local dashboard mirrors (8741/8742) to avoid open redirects.
        allowed = {"http://127.0.0.1:8741/", "http://127.0.0.1:8742/", "http://localhost:8741/", "http://localhost:8742/"}
        base = return_to if return_to.endswith("/") else return_to + "/"
        if base in allowed or return_to in {u.rstrip("/") for u in allowed}:
            dest = return_to.rstrip("/") + "/?ran=1"
    return RedirectResponse(url=dest, status_code=303)


@app.get("/api/demo_run")
def api_demo_run(
    reset: bool = Query(True),
    return_to: str | None = Query(None),
) -> RedirectResponse:
    return demo_run(reset=reset, return_to=return_to)




class SwitchBody(BaseModel):
    powered_on: bool | None = None
    autonomous: bool | None = None
    toggle: bool = False


@app.get("/tools")
def list_tools() -> dict[str, Any]:
    import sys
    tools = Path(__file__).resolve().parents[1] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import registry
    return {"tools": registry.as_dict_list()}


@app.get("/switch")
def get_switch() -> dict[str, Any]:
    from axonforge.switch import SWITCH
    return SWITCH.as_dict()


@app.post("/switch")
def post_switch(body: SwitchBody) -> dict[str, Any]:
    from axonforge.switch import SWITCH
    if body.toggle:
        return SWITCH.toggle()
    if body.autonomous is not None:
        SWITCH.set_autonomous(body.autonomous)
    if body.powered_on is not None:
        SWITCH.set_power(body.powered_on, reason="api")
    return SWITCH.as_dict()



@app.post("/pipeline/plan")
def pipeline_plan(shape_zyx: list[int] | None = None, region: str | None = None) -> dict[str, Any]:
    """Run VoxScout plan pipeline and apply hints into current AxonForge graph."""
    import sys
    tools = Path(__file__).resolve().parents[1] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    from pipeline.flows import plan_pipeline
    from axonforge.bridge import apply_hints_to_graph, load_hints
    shape = tuple(shape_zyx or [64, 64, 64])
    result = plan_pipeline(shape_zyx=shape, region_id=region)
    if RUNTIME.graph is not None:
        result["applied"] = apply_hints_to_graph(RUNTIME.graph, load_hints())
        RUNTIME.graph.save()
    return result


@app.post("/pipeline/close-gaps")
def pipeline_close_gaps(shape_zyx: list[int] | None = None) -> dict[str, Any]:
    import sys
    tools = Path(__file__).resolve().parents[1] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    from pipeline.flows import close_gaps_pipeline
    shape = tuple(shape_zyx or [100, 100, 100])
    return close_gaps_pipeline(shape_zyx=shape)


@app.post("/toolchain/apply-hints")
def toolchain_apply_hints() -> dict[str, Any]:
    from axonforge.bridge import apply_hints_to_graph, load_hints
    if RUNTIME.graph is None:
        return {"ok": False, "error": "no volume submitted"}
    info = apply_hints_to_graph(RUNTIME.graph, load_hints())
    RUNTIME.graph.save()
    return {"ok": True, **info}

if DASH.exists():
    app.mount("/", StaticFiles(directory=str(DASH), html=True), name="dashboard")
