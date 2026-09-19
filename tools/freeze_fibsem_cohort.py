"""Freeze a source-pair-isolated FIB-SEM cohort from the immutable inventory."""
from __future__ import annotations
import argparse, json
from datetime import UTC, datetime
from pathlib import Path

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    inv=json.loads(a.inventory.read_text(encoding='utf-8'))
    pairs=[x for x in inv['pairs'] if x['alignment']=='ALIGNED' and x['modality']=='FIB_SEM']
    pairs.sort(key=lambda x:x['source_pair_id'])
    if len(pairs)!=6: raise ValueError(f'Expected six eligible FIB-SEM pairs, found {len(pairs)}')
    roles=['TRAIN','TRAIN','TRAIN','TRAIN','VALIDATION','TEST']
    records=[]
    for item, role in zip(pairs,roles):
        records.append({'source_pair_id':item['source_pair_id'],'role':role,'dataset':item['dataset'],'modality':'FIB_SEM','resolution_nm_xyz':[8,8,8],
                        'raw':item['raw'],'label':item['label'],'alignment':'ALIGNED','qualification_status':'SOURCE_ALIGNED_EXTERNAL_FIBSEM_INSTANCE_PAIR','source_archive':'datasets/cache/segneuron/EMNeuron-labeled.rar'})
    result={'kind':'FIBSEM_SUPERVISED_COHORT_V1','created_at':datetime.now(UTC).isoformat().replace('+00:00','Z'),'inventory':str(a.inventory.resolve()),
            'split_policy':'whole source volumes are assigned once; no patches cross roles; TEST is frozen before tuning',
            'records':records,'summary':{role:sum(x['role']==role for x in records) for role in ('TRAIN','VALIDATION','TEST')},
            'limitations':['External FIB-SEM source labels; no DVID identity or ground-truth transfer','Source pair isolation is enforced; independent specimen/field metadata beyond source pair IDs is not established here']}
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result['summary']))
    return 0
if __name__=='__main__':raise SystemExit(main())
