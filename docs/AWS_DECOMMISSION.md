# AWS Affinity Decommission

**Status:** Affinity production cloud compute has moved to **Vast.ai**.
AWS must not be required to run Affinity inference.

Hard end state:
- Affinity AWS compute spend = **$0**
- Affinity ASG DesiredCapacity = **0**
- No Affinity EC2 workers running
- No active Affinity scaling
- Cancel/close Affinity quota expansion where AWS allows

This document does **not** authorize deleting scientific artifacts.

## Inventory (read-only)

```bash
# Requires temporary AWS credentials only for cleanup — not for Affinity production.
python tools/audit_aws_affinity_resources.py --profile mloon25 --region us-east-1
```

Expected Affinity-related resources (from prior FAST_002 fleet work):

| Resource | Example / known | Action |
|---|---|---|
| CloudFormation stack | `magaphragma-s7-fast002` | Set ASG desired 0; delete stack only after artifact review |
| Auto Scaling Group | stack-created ASG | `DesiredCapacity=0`, `MinSize=0` |
| EC2 instances | `g6.xlarge` / `g5.xlarge` Name=`magaphragma-s7-fast002` | Terminate / let ASG drain |
| Launch templates | stack-created | Delete with stack |
| DynamoDB claim table | `…MPG4DCAP9MEE` (stack output) | Export nothing scientific; delete with stack after drain |
| S3 artifact bucket | `…wynhdg4uocpn` | **Review before delete** — may hold affinity npy/receipts |
| Quota case | On-Demand G/VT `L-DB2E81BA` case `178973749000673` Desired 64 | Close/cancel in console if still CASE_OPENED |
| Spot quota case | previously CASE_CLOSED | leave closed |

## Scale fleet to zero (safe compute stop)

```bash
python tools/audit_aws_affinity_resources.py \
  --profile mloon25 --region us-east-1 \
  --scale-asg-zero <ASG_NAME> \
  --destroy-confirm SCALE_ASG_TO_ZERO
```

Or AWS CLI:

```bash
aws autoscaling update-auto-scaling-group \
  --auto-scaling-group-name <ASG_NAME> \
  --desired-capacity 0 --min-size 0 \
  --profile mloon25 --region us-east-1
```

## Quota request cancellation / closure

Service Quotas console → EC2 → Running On-Demand G and VT instances → request history.

- If status is **CASE_OPENED** / pending: open the support case and **resolve/close** it as no longer needed (Affinity moved off AWS). Premium Support is not required to leave capacity at 8 vCPU; simply stop pursuing the increase.
- Do **not** open new Affinity GPU quota cases.
- Do **not** close the AWS account.
- Do **not** modify unrelated projects/resources.

Known Affinity case id: `178973749000673` (Desired 64). Leave scientific notes elsewhere; closing the case does not delete local receipts.

## Destructive deletion (explicit only)

Before deleting S3 / DynamoDB / the CloudFormation stack:

1. Run the audit tool and save the JSON inventory under `experiments/phase6e/` as historical evidence if needed.
2. Determine whether the S3 bucket holds **unique** affinity artifacts not already on the local `AFFINITY-FULLVOL-S7-001` tree.
3. If unique: sync/copy to local or cold storage first.
4. Only then delete the stack / bucket with an explicit human confirmation.

Never blind-delete.

## Repository categorization

| Path | Category |
|---|---|
| `infrastructure/aws_s7_workers/*` | KEEP-AS-HISTORICAL-EVIDENCE (marked DECOMMISSIONED) |
| `tools/deploy_s7_aws_fleet_cap2.sh` etc. | REPLACE → refuse stubs |
| `tools/s7_chunk_claim.py` DynamoDB | REPLACE → raise; use local/http |
| `experiments/phase6e/AWS_S7_*` / `AFFINITY_S7_AWS_*` | KEEP-AS-HISTORICAL-EVIDENCE |
| Chunk receipts mentioning AWS workers | KEEP (immutable scientific evidence) |
| Production worker S3 upload | DELETE from production path |

## Production after decommission

Affinity requires **zero AWS services**. Credentials are only needed if you choose to run the optional audit/cleanup tool.
