# ShelfSentry AI - teardown guide (AWS mode)

How to remove the ShelfSentry AI CloudFormation stack while honoring its
deletion and retention policies. This mirrors the README "Cleanup / teardown"
section and adds the per-resource detail for the resources that persist by
policy.

> **This teardown has not been executed from this repo.** The commands below
> are the documented path for *you* to run against your own AWS account. They
> are not verified against live AWS from here. The stack itself
> (`infrastructure/template.yaml`) is authored but not deployed in this
> environment.

## Why teardown is deliberate (the retention policy)

Every **stateful** resource in `infrastructure/template.yaml` is declared with
`DeletionPolicy: Retain` and `UpdateReplacePolicy: Retain` so the stack never
silently destroys your data. Verified in the template:

| Resource (logical id) | Type | Policy |
| --- | --- | --- |
| `MediaBucket` | `AWS::S3::Bucket` | `Retain` / `Retain` |
| `ResultsTable` | `AWS::DynamoDB::Table` | `Retain` / `Retain` |
| `JobsTable` | `AWS::DynamoDB::Table` | `Retain` / `Retain` |
| `ApiLambdaLogGroup` | `AWS::Logs::LogGroup` | `Retain` / `Retain` |
| `WorkerLambdaLogGroup` | `AWS::Logs::LogGroup` | `Retain` / `Retain` |

Consequences for teardown:

1. A `Retain` S3 bucket **cannot be deleted by CloudFormation while it still
   holds objects**, so a `delete-stack` on a non-empty bucket fails. You must
   empty the bucket first.
2. The retained resources above **survive stack deletion** by design. After the
   stack is gone they still exist (and may still incur cost) until you remove
   them individually.

The stateless resources (the two Lambdas, the API Gateway HTTP API + stage +
integration + route, the IAM roles, and the CloudWatch alarm) are deleted
normally when the stack is deleted.

## Prerequisites

- The AWS CLI configured with credentials for the account and region that holds
  the stack (credentials come from the standard provider chain; never embedded).
- The stack name you deployed with. The README uses `shelfsentry-dev`; adjust to
  match yours.
- The resource names, which are stack **outputs**. Fetch them first:

  ```bash
  aws cloudformation describe-stacks --stack-name shelfsentry-dev \
    --query "Stacks[0].Outputs"
  ```

  The outputs you need are `MediaBucketName`, `ResultsTableName`, and
  `JobsTableName`.

## Step 1 - Empty the Retain S3 media bucket FIRST

This is required before the stack can be deleted, and it **deletes data** - make
sure that is what you want. Use the `MediaBucketName` output value:

```bash
aws s3 rm s3://<MediaBucketName output> --recursive
```

If the bucket has versioning or delete markers, `s3 rm --recursive` removes
current objects but may leave noncurrent versions; confirm the bucket is empty:

```bash
aws s3api list-object-versions --bucket <MediaBucketName output> \
  --query "length(Versions || `[]`) + length(DeleteMarkers || `[]`)"
```

If that returns a non-zero count, delete the remaining versions/markers before
proceeding (or empty the bucket from the S3 console, which handles versions).

## Step 2 - Delete the stack

With the bucket emptied, delete the stack. This removes the stateless resources
and *attempts* to remove the retained ones (which it will skip per their
`Retain` policy):

```bash
aws cloudformation delete-stack --stack-name shelfsentry-dev
```

Wait for completion and confirm:

```bash
aws cloudformation wait stack-delete-complete --stack-name shelfsentry-dev
```

If `delete-stack` fails on the bucket, it was not fully emptied - return to
Step 1, confirm the bucket is empty (including versions), then retry.

## Step 3 - Remove the Retain'd resources individually (optional)

The retained resources persist after the stack is gone. Remove each one
explicitly if you want them fully cleaned up. Substitute the output/derived
names (if you did not override the names, the defaults derive from
`${ResourceNamePrefix}-${Environment}-...` with `ResourceNamePrefix=shelfsentry`
and `Environment=dev`).

**The now-empty S3 bucket:**

```bash
aws s3api delete-bucket --bucket <MediaBucketName output>
```

**The DynamoDB tables** (this deletes the stored results, decision traces, and
job records - irreversible):

```bash
aws dynamodb delete-table --table-name <ResultsTableName output>
aws dynamodb delete-table --table-name <JobsTableName output>
```

**The CloudWatch log groups** (named from the prefix/environment, not stack
outputs):

```bash
aws logs delete-log-group --log-group-name "/aws/lambda/shelfsentry-dev-api"
aws logs delete-log-group --log-group-name "/aws/lambda/shelfsentry-dev-worker"
```

> Adjust the log-group names if you deployed with a different
> `ResourceNamePrefix` or `Environment`; the pattern is
> `/aws/lambda/<prefix>-<environment>-api` and
> `/aws/lambda/<prefix>-<environment>-worker`.

## Verify nothing is left (optional)

```bash
aws cloudformation describe-stacks --stack-name shelfsentry-dev   # should error: stack does not exist
aws s3 ls | grep <MediaBucketName output>                          # should return nothing
aws dynamodb list-tables --query "TableNames"                      # should not list the two tables
```

## Local cleanup (no AWS involved)

The local/mock path writes to on-disk directories under `backend/` (for example
`.shelfsentry-local/` from the demo one-liner and `.shelfsentry-evidence/` from
the evidence generator). These are plain local scratch directories; delete them
directly if you want a clean local state. They are unrelated to the AWS stack
and carry no retention policy.
