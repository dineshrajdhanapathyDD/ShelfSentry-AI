# ShelfSentry AI - AWS Deployment Guide

This guide deploys ShelfSentry AI "AWS Mode" to a real AWS account using **one
CloudFormation template** and a **container-image Lambda** that carries the real
OpenCV 5 vision pipeline, the bounded Strands active-perception agent loop, and
the restricted MCP tool surface. No CDK, no Terraform, no SAM transform.

> Honesty note. Some steps run on **your machine**, not inside the Kiro agent
> sandbox. The sandbox cannot build or push container images (no Docker/ECR
> access) and cannot read local files into the AWS API. Those steps are marked
> **[RUN ON YOUR MACHINE]**. Everything the agent could verify against live AWS
> is called out under "What has been verified".

---

## 1. Why container-image Lambda

The worker must `import cv2` (OpenCV 5) and `numpy`. Their native shared
libraries exceed Lambda's 250 MB **unzipped** zip limit, so a classic zip
package cannot carry the real vision pipeline. A **container image** allows up
to **10 GB**, so the real OpenCV 5 code ships intact - no OpenCV 4 substitution,
no mock/placeholder vision path.

Two functions, **one image**:

| Function | Handler (via `ImageConfig.Command`) | Trigger | Timeout |
|---|---|---|---|
| `shelfsentry-<env>-api` | `shelfsentry.awslambda.api.handler` | API Gateway HTTP API | 29 s |
| `shelfsentry-<env>-worker` | `shelfsentry.awslambda.worker.handler` | async invoke from the API (video jobs) | 600 s |

### Lambda vs ECS/Fargate (the 15-minute question)

Lambda caps execution at **15 minutes (900 s)**. The demo workload - a single
image or a short, frame-bounded video (`max_frame_count` default 16, `max_iterations`
default 4) - finishes in seconds to low tens of seconds, well inside 900 s, so
**Lambda is the right fit** and the worker timeout defaults to 600 s. If you later
process long videos that can exceed 15 minutes, **move the worker to ECS/Fargate**
(same container image, run as a Fargate task via an SQS-triggered service). This
template does not raise the worker beyond 900 s, because that ceiling is a hard
Lambda limit, not a tunable.

---

## 2. Prerequisites [RUN ON YOUR MACHINE]

- An AWS account and credentials with permission to create the stack's resources
  (S3, DynamoDB, Lambda, API Gateway, IAM roles, CloudWatch, ECR).
- **Docker** (Docker Desktop on Windows/macOS) with Buildx. The image is
  **arm64**; on Apple Silicon this is native, on x86 Buildx uses QEMU emulation.
- **AWS CLI v2**, configured (`aws configure` or SSO).
- Amazon Bedrock model access for the configured model (see section 6).

Verify:

```bash
aws sts get-caller-identity
docker buildx version
aws --version
```

---

## 3. Build and push the image [RUN ON YOUR MACHINE]

A helper script builds the arm64 image, creates the ECR repo if needed, pushes,
and prints the image URI (by tag and by immutable digest).

Windows (PowerShell):

```powershell
cd "path\to\ShelfSentry AI"
./infrastructure/scripts/build_and_push.ps1 -Region us-east-1 -Repo shelfsentry
```

macOS / Linux:

```bash
cd path/to/ShelfSentry\ AI
./infrastructure/scripts/build_and_push.sh us-east-1 shelfsentry
```

The Dockerfile asserts OpenCV 5 at build time:

```dockerfile
RUN python -c "import cv2, numpy; v=cv2.__version__; \
    assert v.split('.')[0]=='5', 'expected OpenCV 5, got '+v"
```

So a successful build means "OpenCV 5 is importable on arm64 in this image".
The script prints two forms of the image URI:

```
By tag:    <acct>.dkr.ecr.us-east-1.amazonaws.com/shelfsentry:20261022-...
By digest: <acct>.dkr.ecr.us-east-1.amazonaws.com/shelfsentry@sha256:...   <-- prefer this
```

Prefer the **@sha256 digest** for `EcrImageUri`: it is immutable, so the stack
deploys the exact bytes you built.

> Do not deploy the stack until the image exists in ECR. The functions reference
> `EcrImageUri`; CloudFormation will fail to create them if the image is absent.

---

## 4. Deploy the stack [RUN ON YOUR MACHINE]

```bash
aws cloudformation deploy \
  --template-file infrastructure/template.yaml \
  --stack-name shelfsentry-dev \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
      EcrImageUri=<acct>.dkr.ecr.us-east-1.amazonaws.com/shelfsentry@sha256:... \
      Environment=dev \
      BedrockModelId=amazon.nova-lite-v1:0 \
  --region us-east-1
```

`CAPABILITY_NAMED_IAM` is required because the two execution roles have explicit
names. Review the change set first if you prefer:

```bash
aws cloudformation deploy ... --no-execute-changeset   # prints the change set ARN
aws cloudformation describe-change-set --change-set-name <arn> --region us-east-1
```

### Parameters

| Parameter | Default | Notes |
|---|---|---|
| `EcrImageUri` | (required) | Full ECR URI; prefer the `@sha256` digest form. |
| `Environment` | `dev` | `dev` / `staging` / `prod`; used in resource names. |
| `ResourceNamePrefix` | `shelfsentry` | Name prefix. |
| `BedrockModelId` | `amazon.nova-lite-v1:0` | Any Converse-capable text model. |
| `MediaUploadPrefix` | `inspections/` | S3 key prefix for uploads. |
| `MediaRetentionDays` | `90` | Lifecycle expiry for media objects. |
| `ApiLambdaMemoryMb` / `ApiLambdaTimeoutSeconds` | `512` / `29` | API function size. |
| `WorkerLambdaMemoryMb` / `WorkerLambdaTimeoutSeconds` | `2048` / `600` | Worker size; timeout <= 900. |
| `LogRetentionDays` | `30` | CloudWatch log retention. |

### Read the outputs

```bash
aws cloudformation describe-stacks --stack-name shelfsentry-dev \
  --query "Stacks[0].Outputs" --region us-east-1
```

Key outputs: `ApiUrl`, `MediaBucketName`, `ResultsTableName`, `ApiLambdaName`,
`WorkerLambdaName`, `BedrockModelId`.

---

## 5. Validate end to end

### 5.1 Health

```bash
API_URL=$(aws cloudformation describe-stacks --stack-name shelfsentry-dev \
  --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" --output text --region us-east-1)
curl "$API_URL/health"
# -> {"apiVersion":"v1","mode":"aws","status":"ok","bedrockModelId":"amazon.nova-lite-v1:0"}
```

### 5.2 Real image inspection (synchronous)

Send a real shelf image (base64) and read back a real report with a decision
trace. This exercises the REAL OpenCV 5 pipeline + the bounded agent loop with
Bedrock as the primary decision path:

```bash
B64=$(base64 -w0 your-shelf.jpg)   # macOS: base64 -i your-shelf.jpg
curl -X POST "$API_URL/inspections" \
  -H "content-type: application/json" \
  -d "{\"content_type\":\"image/jpeg\",\"media_base64\":\"$B64\"}"
```

Expect a `mode:"aws"` envelope whose `data` is an `InspectionReport` with a
non-empty `decision_trace`. If Bedrock is reachable, trace decisions show
`decided_by:"bedrock"`; if Bedrock is unavailable the loop records
`used_fallback:true` with a reason (the deterministic policy) - still a real,
finding-derived decision, never a fabricated result.

### 5.3 Video inspection (asynchronous)

```bash
B64=$(base64 -w0 your-clip.mp4)
curl -X POST "$API_URL/inspections" \
  -H "content-type: application/json" \
  -d "{\"content_type\":\"video/mp4\",\"media_base64\":\"$B64\"}"
# -> 202 with {"data":{"job_id":"job-...","status":"queued", ...}}
```

Poll the job:

```bash
curl "$API_URL/jobs/job-..."
# status transitions queued -> running -> succeeded | failed
```

Then fetch the report by its inspection id:

```bash
curl "$API_URL/inspections/insp-..."
```

### 5.4 Logs

```bash
aws logs tail /aws/lambda/shelfsentry-dev-worker --follow --region us-east-1
aws logs tail /aws/lambda/shelfsentry-dev-api --follow --region us-east-1
```

---

## 6. Amazon Bedrock model access

The agent's primary decision path calls Bedrock through the **Converse API**, so
any Converse-capable text model works; the model is pure configuration
(`BedrockModelId`), never a secret.

- Default is **`amazon.nova-lite-v1:0`** (Amazon Nova Lite), chosen because it is
  available on-demand in the verified account/region and returns the agent's
  JSON action shape via Converse.
- If you prefer Anthropic Claude, request model access in the Bedrock console and
  pass that model id (or its inference-profile id) as `BedrockModelId`; no code
  change is needed because `BedrockDecisionModel` already uses Converse.

The worker role grants `bedrock:InvokeModel` scoped to the single configured
model ARN only.

---

## 7. Security and cost

Security (enforced by the template and the code):

- S3 bucket is private, encrypted (AES256), public-access-blocked, versioned,
  and denies non-TLS access; media is addressed only by server-minted opaque
  `media://` refs (never a caller path/URL).
- IAM is least-privilege per function: each role is scoped to its own log group,
  the media prefix, the single DynamoDB table, and - for the API - async invoke
  of the worker only; the worker additionally gets `bedrock:InvokeModel` on one
  model ARN. No wildcard admin, no embedded secrets.
- Credentials come from the Lambda execution role via the standard provider
  chain; nothing is read into application context.

Cost (us-east-1 order-of-magnitude for light demo use):

- Lambda (arm64): a few cents per thousand short inspections; image storage in
  ECR is a few cents/GB-month.
- DynamoDB PAY_PER_REQUEST and S3: pennies at demo volume; media expires after
  `MediaRetentionDays`.
- Bedrock Nova Lite: fractions of a cent per inspection (tens of input tokens,
  ~20 output tokens per decision).
- The account's default Lambda concurrency may be low (e.g. 10); fine for a demo
  (API + worker), raise via Service Quotas for load.

---

## 8. Teardown

```bash
# 1. Empty the media bucket (it is Retain-on-delete, so it blocks stack delete).
aws s3 rm s3://<MediaBucketName> --recursive --region us-east-1

# 2. Delete the stack.
aws cloudformation delete-stack --stack-name shelfsentry-dev --region us-east-1
aws cloudformation wait stack-delete-complete --stack-name shelfsentry-dev --region us-east-1
```

Retain-policy resources (S3 bucket, DynamoDB table, log groups) survive a stack
delete by design (Req 20.3). Remove them manually if you want them gone:

```bash
aws s3 rb s3://<MediaBucketName> --region us-east-1
aws dynamodb delete-table --table-name <ResultsTableName> --region us-east-1
```

Optionally delete the ECR repo and its images:

```bash
aws ecr delete-repository --repository-name shelfsentry --force --region us-east-1
```

---

## 9. What has been verified vs what you run

Verified live by the agent against account 466742534146 / us-east-1:

- The CloudFormation template passes `aws cloudformation validate-template`
  (recognized 13 parameters; requires `CAPABILITY_NAMED_IAM`).
- Amazon Nova Lite (`amazon.nova-lite-v1:0`) is available and returns the agent's
  exact JSON action shape via the Converse API.
- The `opencv-python` aarch64 manylinux wheel is published (so the arm64 image
  build resolves OpenCV 5), and `BedrockDecisionModel` already uses Converse
  (so Nova needs zero code change).
- The Lambda transport handlers pass a dedicated test suite (routing, status
  mapping, base64, async video handoff) with the full backend suite green.

Run by you on your machine (cannot be done from the agent sandbox):

- `docker buildx build ... --push` to publish the image to ECR.
- `aws cloudformation deploy ...` to create the stack.
- The end-to-end `curl` validation in section 5 against the live `ApiUrl`.

When you run those steps, capture the `ApiUrl`, a real inspection response, and
the job transitions; paste any failures here and they will be diagnosed against
the live stack.
