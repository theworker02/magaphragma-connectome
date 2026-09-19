# Blocker resolution ledger

## BR-001 — WASPSYN artifact unavailable

- **Status:** EXTERNAL_BLOCKER
- **Why it blocks:** No downloadable source archive can be registered or inspected.
- **Actions attempted:** Publication-referenced CodaLab competition `9169` opened through direct and visual access; both returned HTTP 502 on 2026-09-16.
- **Required evidence:** Official artifact URL, file identity, and terms applying to that artifact.
- **Next action:** Recheck official service or obtain author-hosted distribution; do not use an unofficial mirror without attribution.

## BR-002 — CATMAID source export

- **Status:** RESOLVED for connector-link skeleton identity
- **Resolution:** Complete row-schema inspection and public treenode detail testing established that each exported row already contains `skeleton_id` and `treenode_id`; the public skeleton compact-detail endpoint returns morphology.
- **Actions attempted:** Public project, stack, annotation, relation-type, and connector bounding-box APIs. Acquired a deterministic local export of 28,775 connector-link rows / 5,821 unique connectors.
- **Artifact:** `datasets/cache/catmaid-lamina-export/connectors-bounds-nm.json` (local only), SHA-256 `168fd38b723ca52a8bd43237643e31d79e2d6aa7d2d0369bfc5d7790c291762e`.
- **Coordinate finding:** connector XYZ are physical nanometres; stack voxel coordinates require division by its verified 8 nm resolution only within this CATMAID stack frame.
- **Next action:** Public `GET /1/skeletons/{id}/compact-detail` is used only through the documented waspem-lamina endpoint, with a resumable, rate-limited local cache. No authentication bypass is permitted.

## BR-003 — CATMAID data rights

- **Status:** OPEN
- **Why it blocks release:** Public readability and article availability do not establish data redistribution rights.
- **Allowed now:** Local, read-only, provenance-preserving analysis of publicly returned records; no redistribution or scientific release.
- **Next action:** Obtain a source rights statement for data/API export scope.

## BR-004 — Cross-frame relation to Phase 2 DVID

- **Status:** UNKNOWN
- **Why it blocks:** The DVID and CATMAID datasets cannot be assumed to be the same specimen or coordinate frame.
- **Resolution:** Not needed for CATMAID-local analysis. Cross-registration remains prohibited until independently supported.
