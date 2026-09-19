# G3 interface-level supervision contract

G3 review questions are organized as raw-EM-only `INTERFACE_ID` groups. Each
member is an explicitly reviewed, direct Z/Y/X affinity pair; a membrane-like
raw feature merely selects the group and never supplies its biological label.

The append-only chain is:

`INTERFACE_ID → question ID → reviewer decision → direct affinity pair`.

Pair-level targets give the affinity loss local spatial support around one
candidate interface. The interface remains the statistical evidence unit:
future G3 materialization must report both pair and interface counts, reject
contradictory decisions within an interface, and cap or normalize loss weight
so a many-member interface cannot dominate fitting.

`MV-GTVOL-000004` is excluded from G3 region selection, review, training,
validation selection, and every interface queue.
