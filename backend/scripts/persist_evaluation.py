"""Run the real evaluation and persist the measured result to DynamoDB.

Runs shelfsentry.eval.run_evaluation() - the REAL local pipeline over the
synthetic labeled set - and writes the resulting measured-only EvaluationResult
into the deployed single table via AwsStorageAdapter, so the frontend's
Evaluation page (GET /evaluations) shows genuine measured numbers.

Nothing here fabricates a metric: the record is exactly what the runner
measured, stamped is_demo=True (synthetic labels) and measured_only=True.

Usage (reads non-secret resource names from env; AWS creds via the standard
provider chain):

    SHELFSENTRY_S3_BUCKET=<MediaBucketName> \
    SHELFSENTRY_DDB_TABLE=<ResultsTableName> \
    SHELFSENTRY_AWS_REGION=us-east-1 \
    python scripts/persist_evaluation.py

Or pass --bucket/--table/--region explicitly.
"""

from __future__ import annotations

import argparse
import os
import sys

from shelfsentry.data.aws_storage import AwsStorageAdapter
from shelfsentry.eval import run_evaluation


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", default=os.environ.get("SHELFSENTRY_S3_BUCKET"))
    ap.add_argument("--table", default=os.environ.get("SHELFSENTRY_DDB_TABLE"))
    ap.add_argument("--region", default=os.environ.get("SHELFSENTRY_AWS_REGION", "us-east-1"))
    args = ap.parse_args()

    if not args.bucket or not args.table:
        print(
            "ERROR: set SHELFSENTRY_S3_BUCKET and SHELFSENTRY_DDB_TABLE "
            "(or pass --bucket/--table). These are the stack's MediaBucketName "
            "and ResultsTableName outputs.",
            file=sys.stderr,
        )
        return 2

    print("Running the real evaluation over the synthetic labeled set...")
    result = run_evaluation(is_demo=True)  # synthetic labels -> is_demo=True
    print(
        f"Measured: run={result.eval_run_id} samples={result.sample_count} "
        f"FP={len(result.false_positives)} FN={len(result.false_negatives)} "
        f"decision_correctness={result.agent_decision_correctness} "
        f"p50={result.latency_ms_p50}ms p95={result.latency_ms_p95}ms"
    )

    adapter = AwsStorageAdapter(bucket=args.bucket, table=args.table, region=args.region)
    adapter.save_eval(result)
    print(f"Persisted EvaluationResult {result.eval_run_id} to table {args.table}.")

    # Read back to confirm it is retrievable (and not fabricated on read).
    stored = adapter.list_evals()
    print(f"Table now holds {len(stored)} evaluation record(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())