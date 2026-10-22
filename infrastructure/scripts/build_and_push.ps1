<#
.SYNOPSIS
  Build the ShelfSentry AI container image (arm64) and push it to Amazon ECR.

.DESCRIPTION
  Run this ON YOUR MACHINE (Docker Desktop required). The Kiro sandbox cannot
  build or push container images, so this is a manual, reproducible step. After
  it completes it prints the EcrImageUri to pass to the CloudFormation deploy.

  The image is arm64 (Graviton). If you are on an x86 machine, Docker Buildx with
  QEMU emulation builds arm64 fine (slower). On Apple Silicon it is native.

.PARAMETER Region
  AWS region (default us-east-1). Must match where you deploy the stack.

.PARAMETER Repo
  ECR repository name (default shelfsentry).

.PARAMETER Tag
  Image tag (default: a UTC timestamp, so each build is uniquely addressable).

.EXAMPLE
  ./build_and_push.ps1 -Region us-east-1 -Repo shelfsentry
#>
param(
  [string]$Region = "us-east-1",
  [string]$Repo   = "shelfsentry",
  [string]$Tag    = (Get-Date -Format "yyyyMMdd-HHmmss")
)

$ErrorActionPreference = "Stop"

# Resolve paths relative to this script: build context is backend/.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root      = Resolve-Path (Join-Path $ScriptDir "..\..")
$Backend   = Join-Path $Root "backend"
$Dockerfile = Join-Path $Root "infrastructure\docker\Dockerfile"

Write-Host "==> Account / region"
$AccountId = (aws sts get-caller-identity --query Account --output text).Trim()
if (-not $AccountId) { throw "Could not resolve AWS account id. Run 'aws configure' or set credentials." }
$Registry = "$AccountId.dkr.ecr.$Region.amazonaws.com"
$ImageUri = "$Registry/${Repo}:$Tag"
Write-Host "    account=$AccountId region=$Region"
Write-Host "    image=$ImageUri"

Write-Host "==> Ensure ECR repository exists (idempotent)"
aws ecr describe-repositories --repository-names $Repo --region $Region *> $null
if ($LASTEXITCODE -ne 0) {
  aws ecr create-repository `
    --repository-name $Repo `
    --image-scanning-configuration scanOnPush=true `
    --image-tag-mutability IMMUTABLE `
    --region $Region | Out-Null
  Write-Host "    created repository $Repo"
} else {
  Write-Host "    repository $Repo already exists"
}

Write-Host "==> Docker login to ECR"
aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $Registry
if ($LASTEXITCODE -ne 0) { throw "docker login failed" }

Write-Host "==> Build arm64 image (context: $Backend)"
# --provenance=false keeps the pushed artifact a single-arch image manifest that
# Lambda accepts. --platform linux/arm64 matches the function Architecture.
docker buildx build `
  --platform linux/arm64 `
  --provenance=false `
  -f $Dockerfile `
  -t $ImageUri `
  --push `
  $Backend
if ($LASTEXITCODE -ne 0) { throw "docker build/push failed" }

Write-Host "==> Resolve the immutable digest (recommended for CloudFormation)"
$Digest = (aws ecr describe-images `
  --repository-name $Repo `
  --image-ids imageTag=$Tag `
  --region $Region `
  --query "imageDetails[0].imageDigest" --output text).Trim()
$ImageByDigest = "$Registry/$Repo@$Digest"

Write-Host ""
Write-Host "================ IMAGE PUBLISHED ================"
Write-Host "By tag:    $ImageUri"
Write-Host "By digest: $ImageByDigest   <-- prefer this for EcrImageUri"
Write-Host "================================================"
Write-Host ""
Write-Host "Next: deploy with"
Write-Host "  aws cloudformation deploy ``"
Write-Host "    --template-file infrastructure/template.yaml ``"
Write-Host "    --stack-name shelfsentry-dev ``"
Write-Host "    --capabilities CAPABILITY_NAMED_IAM ``"
Write-Host "    --parameter-overrides EcrImageUri=$ImageByDigest ``"
Write-Host "    --region $Region"
