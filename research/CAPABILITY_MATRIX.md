# Capability matrix

| Operation | Dataset/frame | Evidence | Rights | Status | Allowed action |
| --- | --- | --- | --- | --- | --- |
| Real voxel viewing and machine segmentation | Phase 2 DVID frame | Real local raw cache | No redistribution/release basis | PASS, local-only | Ingest, inspect, segment; no scientific release |
| CATMAID public metadata/annotations | CATMAID project 1 | Documented public GET endpoints | Export rights unresolved | PASS, local audit | Read, hash public responses |
| CATMAID connector locations/relation IDs | CATMAID project 1 physical-nm frame | 28,775 acquired source rows | Export rights unresolved | PASS, local-only | Validate coordinate frame; no graph |
| CATMAID skeleton/partner mapping | CATMAID project 1 | Requires session-enabled documented endpoint | Not yet established | GATED | Needs user-authorized session/source export |
| WASPSYN artifact import | WASPSYN local frame | Publication specification only | CC-BY reported, artifact unavailable | EXTERNAL_BLOCKER | Wait for official artifact endpoint |
| CATMAID ↔ DVID transform | Cross-frame | No shared specimen/parent lineage, source transform, or independent landmarks | N/A | INSUFFICIENT_EVIDENCE | CATMAID-derived seeds and physical mappings prohibited |
| DVID-native physical reconstruction | `MV-PHASE5-DOMAIN-001` | Real local raw crop, 8 overlapping chunks | Local processing only | ELIGIBLE_AWAITING_REGISTERED_BACKEND | Must retain `SPECIMEN_UNKNOWN` unless source lineage changes |
