# SegNeuron FRMC execution gate

## Attempted environment

The checked-out official SegNeuron source pins its original FRMC postprocessor
to Linux/WSL with conda-forge `python-elf=0.8.1`, `nifty`, and `vigra`.
The source explicitly warns that ELF 0.9 changes the C++ backend and that a
successful generic package installation is not proof that FRMC works.

## Initial host result — 2026-09-16

- `wsl.exe --status` and `wsl.exe -l -v`: failed with
  `Wsl/EnumerateDistros/Service/E_ACCESSDENIED`.
- The project-local SegNeuron environment failed the official backend import:
  `ModuleNotFoundError: No module named 'elf'`.

At that time, `FRMC_POSTPROCESSING = UNAVAILABLE_WSL_SERVICE_ACCESS_DENIED`.
No substitute is labeled FRMC. The affinity-component pseudolabel method is a
separate, explicitly non-production proposal heuristic and its failure is
retained for review.

## Next executable action

Restore authorized WSL service access or provide a functioning Linux/conda
environment, then create the official `segneuron-postprocess` environment from
`third_party/segneuron/environment-postprocess.yml`, prove the three ELF imports,
and run `Postprocess/FRMC_post.py` against the retained zero-shot affinity and
boundary outputs. This still produces only `MACHINE_PSEUDOLABEL` until
independently adjudicated DVID-native labels exist.

## Transient installation execution block — 2026-09-16

```ini
INSTALL_ATTEMPT = BLOCKED_BEFORE_EXECUTION
BLOCKER = APPROVAL_SERVICE_MODEL_CAPACITY
USER_APPROVAL_PROMPT = NOT_PRESENTED
MICROMAMBA_INSTALL = NOT_STARTED
ENVIRONMENT_MUTATION = NONE
SCIENTIFIC_STATE = UNCHANGED
CLASSIFICATION = EXECUTION_BLOCKED_EXTERNAL_CAPACITY
```

The approval/execution service rejected the scoped Micromamba command before
it was started, twice. This is not `INSTALL_FAILED`, does not imply a broken
package, and does not justify redesigning or downgrading the FRMC pipeline.
The next action remained a retry of the same local Ubuntu-user Micromamba
installation when the external service accepted the operation.

## Micromamba bootstrap result — 2026-09-16

The subsequent scoped Ubuntu-user installation executed successfully:

```ini
MICROMAMBA_INSTALL = PASS
MICROMAMBA_PATH = ~/.local/bin/micromamba
MICROMAMBA_VERSION = 2.9.0
ENVIRONMENT_MUTATION = MICROMAMBA_BINARY_ONLY
SCIENTIFIC_STATE = UNCHANGED
```

No SegNeuron postprocessing environment or FRMC execution was created by that
step.

## Postprocessing-environment execution block — 2026-09-16

```ini
INSTALL_ATTEMPT = BLOCKED_BEFORE_EXECUTION
TARGET = segneuron-postprocess
BLOCKER = APPROVAL_SERVICE_MODEL_CAPACITY
USER_APPROVAL_PROMPT = NOT_PRESENTED
ENVIRONMENT_MUTATION = NONE
SCIENTIFIC_STATE = UNCHANGED
CLASSIFICATION = EXECUTION_BLOCKED_EXTERNAL_CAPACITY
```

The approval/execution service rejected the environment-creation command
before it started. This is not `INSTALL_FAILED`; it provides no evidence about
the official dependency set or FRMC itself. The next executable action is to
retry this same isolated environment creation when the service accepts it.

## Official FRMC environment and execution — 2026-09-16

The corrected WSL command executed after eliminating PowerShell expansion of
the POSIX home path. The preceding zero-exit attempts using the expanded path
did not create an environment or output artifact and are not treated as
execution evidence.

```ini
FRMC_ENVIRONMENT = PASS
ENVIRONMENT_PREFIX = /home/mloon25/.local/share/micromamba/envs/segneuron-postprocess
PYTHON = 3.10.21
PYTHON_ELF = 0.8.1
NUMPY = 1.26.4
NIFTY = 1.2.4
VIGRA = 1.12.3
FRMC_ELF_IMPORTS = PASS
```

The checked-out `Postprocess/FRMC_post.py` then ran with real SegNeuron
zero-shot affinities and boundaries from `MV-GTVOL-000004` (real DVID EM),
with `beta=0.25` and **without** a ground-truth input. It persisted
`local_research_build/phase5e-a/frmc/MV-GTVOL-000004-frmc-labels_zyx.npy`:

```ini
FRMC_EXECUTION = PASS
OUTPUT_SHA256 = dbc332d3701496ce5c78b2a93a3b15ab75d89cadfcd14db1431d36dfc84f67a7
OUTPUT_SHAPE_ZYX = 192,192,192
OUTPUT_DTYPE = uint32
OUTPUT_INSTANCES = 1
OUTPUT_OCCUPANCY = 1.0
OUTPUT_STATUS = MACHINE_PSEUDOLABEL
SCIENTIFIC_STATE = UNCHANGED
```

The append-only audit at
`local_research_build/phase5e-a/frmc/MV-GTVOL-000004-frmc-severity-audit.json`
flags the sole component as `NEAR_FULL_VOLUME_MERGE_CANDIDATE`. The artifact
is retained as reproducible negative proposal evidence; it is prohibited from
annotation, ground truth, held-out metrics, `MV-SEG`, and production use.
