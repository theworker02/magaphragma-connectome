#!/usr/bin/env python3
"""Audit Affinity-related AWS resources (read-only by default).

Does NOT delete scientific data. Destructive cleanup requires --destroy-confirm TOKEN
after inventory review. See docs/AWS_DECOMMISSION.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def audit(profile: str | None, region: str) -> dict:
    try:
        import boto3
    except ImportError:
        return {
            "ok": False,
            "error": "boto3 not installed — cannot audit. Install temporarily or use AWS console.",
            "at": _now(),
        }

    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    out: dict = {"at": _now(), "region": region, "profile": profile, "resources": {}}

    # EC2 instances tagged affinity / s7
    ec2 = session.client("ec2", region_name=region)
    try:
        pages = ec2.get_paginator("describe_instances").paginate(
            Filters=[{"Name": "tag:Name", "Values": ["*s7*", "*affinity*", "*magaphragma*"]}]
        )
        instances = []
        for page in pages:
            for res in page.get("Reservations", []):
                for inst in res.get("Instances", []):
                    instances.append(
                        {
                            "InstanceId": inst.get("InstanceId"),
                            "State": (inst.get("State") or {}).get("Name"),
                            "InstanceType": inst.get("InstanceType"),
                            "Tags": {t["Key"]: t["Value"] for t in inst.get("Tags", [])},
                        }
                    )
        out["resources"]["ec2_instances"] = instances
    except Exception as e:
        out["resources"]["ec2_instances_error"] = str(e)

    # ASG
    try:
        asg = session.client("autoscaling", region_name=region)
        groups = asg.describe_auto_scaling_groups().get("AutoScalingGroups", [])
        out["resources"]["auto_scaling_groups"] = [
            {
                "AutoScalingGroupName": g.get("AutoScalingGroupName"),
                "DesiredCapacity": g.get("DesiredCapacity"),
                "MinSize": g.get("MinSize"),
                "MaxSize": g.get("MaxSize"),
                "Instances": len(g.get("Instances") or []),
            }
            for g in groups
            if "s7" in (g.get("AutoScalingGroupName") or "").lower()
            or "magaphragma" in (g.get("AutoScalingGroupName") or "").lower()
            or "affinity" in (g.get("AutoScalingGroupName") or "").lower()
        ]
    except Exception as e:
        out["resources"]["asg_error"] = str(e)

    # CloudFormation
    try:
        cfn = session.client("cloudformation", region_name=region)
        stacks = cfn.list_stacks(StackStatusFilter=[
            "CREATE_COMPLETE", "UPDATE_COMPLETE", "UPDATE_ROLLBACK_COMPLETE",
            "CREATE_IN_PROGRESS", "UPDATE_IN_PROGRESS",
        ]).get("StackSummaries", [])
        out["resources"]["cloudformation_stacks"] = [
            {"StackName": s.get("StackName"), "StackStatus": s.get("StackStatus")}
            for s in stacks
            if "s7" in (s.get("StackName") or "").lower()
            or "magaphragma" in (s.get("StackName") or "").lower()
            or "affinity" in (s.get("StackName") or "").lower()
        ]
    except Exception as e:
        out["resources"]["cfn_error"] = str(e)

    # DynamoDB
    try:
        ddb = session.client("dynamodb", region_name=region)
        tables = ddb.list_tables().get("TableNames", [])
        out["resources"]["dynamodb_tables"] = [t for t in tables if "s7" in t.lower() or "claim" in t.lower() or "affinity" in t.lower()]
    except Exception as e:
        out["resources"]["ddb_error"] = str(e)

    # S3 buckets (name filter only — do not list objects unless asked)
    try:
        s3 = session.client("s3", region_name=region)
        buckets = s3.list_buckets().get("Buckets", [])
        out["resources"]["s3_buckets"] = [
            b["Name"] for b in buckets
            if any(x in b["Name"].lower() for x in ("s7", "affinity", "magaphragma", "wynhdg4uocpn"))
        ]
    except Exception as e:
        out["resources"]["s3_error"] = str(e)

    # Quota requests
    try:
        sq = session.client("service-quotas", region_name=region)
        hist = sq.list_requested_service_quota_change_history_by_quota(
            ServiceCode="ec2", QuotaCode="L-DB2E81BA"
        ).get("RequestedQuotas", [])
        out["resources"]["quota_g_vt_ondemand"] = [
            {
                "Status": h.get("Status"),
                "DesiredValue": h.get("DesiredValue"),
                "CaseId": h.get("CaseId"),
                "Created": str(h.get("Created")),
            }
            for h in hist[:5]
        ]
    except Exception as e:
        out["resources"]["quota_error"] = str(e)

    out["ok"] = True
    out["destructive"] = False
    out["note"] = (
        "Read-only inventory. To scale ASG to 0 or delete stacks, follow docs/AWS_DECOMMISSION.md "
        "and re-run with explicit confirmation. Do not delete S3 scientific artifacts without review."
    )
    return out


def scale_asg_to_zero(profile: str | None, region: str, name: str, confirm: str) -> dict:
    if confirm != "SCALE_ASG_TO_ZERO":
        return {"ok": False, "error": "pass --destroy-confirm SCALE_ASG_TO_ZERO"}
    import boto3

    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    asg = session.client("autoscaling", region_name=region)
    asg.update_auto_scaling_group(AutoScalingGroupName=name, DesiredCapacity=0, MinSize=0)
    return {"ok": True, "action": "DesiredCapacity=0", "asg": name}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    ap.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    ap.add_argument("--scale-asg-zero", default=None, help="ASG name to set DesiredCapacity=0")
    ap.add_argument("--destroy-confirm", default="")
    args = ap.parse_args()
    if args.scale_asg_zero:
        print(json.dumps(scale_asg_to_zero(args.profile, args.region, args.scale_asg_zero, args.destroy_confirm), indent=2))
        return 0
    print(json.dumps(audit(args.profile, args.region), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
