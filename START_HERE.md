# Start here

## Scientific guardrails

Do not enter a neuron, synapse, segment, region boundary, or connection merely to exercise the software. Test material belongs under `fixtures/synthetic/` and must be visibly marked synthetic. The released biological registry starts empty by design.

## First milestone workflow

1. Review [research/DATA_SOURCES.md](research/DATA_SOURCES.md) and obtain data using the documented owner portal and terms.
2. Record the exact artifact URL, license text/version, file checksum, and access date in `datasets/registry.json` before enabling its download.
3. Run `vigilia ingest` with a signed/authoritative metadata sidecar.
4. Add an immutable dataset manifest and a coordinate-validation experiment.
5. Import only license-permitted published annotation, preserving original identifiers and citations.
6. Compare a small lamina target with the published result before extending any reconstruction.

## Developer commands

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
vigilia sources
vigilia validate-manifest datasets/registry.json
vigilia verify
vigilia serve --port 8765
```

## Data policy

- Never commit EM imagery, derived volume chunks, or large binary reconstructions to ordinary Git.
- The downloader only operates on a source explicitly marked `download_approved` with SHA-256 metadata.
- A landing page is not a direct download URL.
- No raw image may be redistributed unless its source rights explicitly permit redistribution.


## Engineering toolchain (AxonForge + tools)

All registered tools are wired into the Connectome project (not CLI-only orphans).

```powershell
python -m pip install -e .
vigilia toolchain status
vigilia toolchain doctor
vigilia toolchain pipeline plan --shape 64,64,64
vigilia toolchain pipeline close-gaps --shape 64,64,64
vigilia toolchain pipeline reconcile
vigilia toolchain pipeline audit
vigilia toolchain pipeline connectivity
vigilia toolchain pipeline diff
vigilia toolchain pipeline mass-status
connectome status
```

AxonForge consumes VoxScout hints, NeuroCache, TraceWire, and post-run GapHound / MorphGuard / SeamSmith+BranchJudge / SynapseLens / DeltaGraph / EdgeProbe / HyperDrain / ArtifactVet / RunDoctor via `axonforge.bridge`. Dashboard: `http://127.0.0.1:8741/` (API) / `:8742` (static). See [docs/toolchain.md](docs/toolchain.md).

