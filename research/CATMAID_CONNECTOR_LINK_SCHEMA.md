# CATMAID connector-link schema

The immutable public bounding-box export has 28,775 rows with 11 fields:

`[connector_id, x_nm, y_nm, z_nm, skeleton_id, confidence, creator_id, treenode_id, creation_time, edition_time, relation_id]`

This was verified against CATMAID's documented `/{project}/treenodes/{id}/compact-detail` endpoint: export treenode `329991` resolves to skeleton `216561`. The documented `/{project}/skeletons/{id}/compact-detail` endpoint also returned its source morphology.

Relation IDs are source metadata: `15 = presynaptic_to`, `16 = postsynaptic_to`. Connector coordinates are physical nanometres in CATMAID's project frame. This schema permits legitimate connector → treenode → skeleton reconstruction without spatial identity inference.
