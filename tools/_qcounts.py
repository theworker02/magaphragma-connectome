import json
from collections import Counter
from pathlib import Path
p = Path("/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome/experiments/phase6e/AFFINITY-FULLVOL-S7-001/queue_state.json")
c = json.loads(p.read_text())
print(dict(Counter(c["status_by_id"].values())))
