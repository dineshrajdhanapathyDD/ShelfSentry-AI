"""CloudFormation template validation and security-constraint tests (task 15.2).

Validates ``infrastructure/template.yaml`` offline and asserts the storage /
IAM / observability security constraints the design requires. These tests run
WITHOUT network access or AWS credentials: the parse, structural, and security
assertions work on the local file, and the optional cfn-lint check uses the
cfn-lint Python API against a single standard region (us-east-1).

Covered requirements:

* Req 24.3 - a test that validates the CloudFormation template (parse +
  structure, plus the cfn-lint API error-level gate when cfn-lint is
  importable).
* Req 24.4 - tests verifying IAM and storage security constraints.
* Req 18.2 - S3 bucket private, encrypted, public-access-blocked.
* Req 19.2 - every Lambda has explicit Timeout and MemorySize.
* Req 19.3 - at least one log group with retention and at least one alarm.
* Req 20.1 - no IAM statement grants Action "*" on Resource "*".
* Req 20.2 - no embedded secrets.
* Req 20.3 - stateful resources (S3, DynamoDB) use Retain policies.

The CloudFormation short-form intrinsic tags (``!Ref``, ``!GetAtt``, ``!Sub``,
``!If`` ...) are not valid plain YAML, so a bespoke ``SafeLoader`` registers
multi-constructors that turn every ``!Tag`` into a tagged dict
(``{"Fn::Tag": value}`` / ``{"Ref": value}``). The loader is intentionally
small and local to this module.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import pytest
import yaml

# ---------------------------------------------------------------------------
# Template location
# ---------------------------------------------------------------------------
# tests/infrastructure/test_template.py -> repo root is three parents up from
# the "backend" dir. Resolve the single infrastructure definition robustly so
# the test does not depend on the pytest invocation directory.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]  # .../ShelfSentry AI
TEMPLATE_PATH = _REPO_ROOT / "infrastructure" / "template.yaml"


# ---------------------------------------------------------------------------
# A small CloudFormation-aware YAML loader (local to this test module).
# ---------------------------------------------------------------------------
class _CfnLoader(yaml.SafeLoader):
    """SafeLoader that understands CloudFormation short-form intrinsics.

    A plain ``yaml.safe_load`` raises on tags such as ``!Ref`` / ``!GetAtt``.
    We register multi-constructors for the ``!`` prefix so every intrinsic is
    materialized as its long-form single-key mapping, e.g.::

        !Ref Foo            -> {"Ref": "Foo"}
        !GetAtt A.Arn       -> {"Fn::GetAtt": "A.Arn"}
        !Sub "x-${Y}"       -> {"Fn::Sub": "x-${Y}"}
        !If [C, a, b]       -> {"Fn::If": [C, a, b]}

    ``Ref`` keeps its bare name; every other tag is namespaced under ``Fn::``,
    matching how CloudFormation expands the short form. This is enough for the
    structural and security assertions below (we never need to *evaluate* the
    intrinsics, only to read the template shape and scan it for strings).
    """


def _cfn_multi_constructor(loader: _CfnLoader, tag_suffix: str, node: yaml.Node) -> Any:
    # ``Ref`` is the one intrinsic that is not namespaced under ``Fn::``.
    key = "Ref" if tag_suffix == "Ref" else f"Fn::{tag_suffix}"

    if isinstance(node, yaml.ScalarNode):
        value: Any = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:  # MappingNode
        value = loader.construct_mapping(node, deep=True)

    return {key: value}


# ``!Tag`` short forms all share the ``!`` prefix.
_CfnLoader.add_multi_constructor("!", _cfn_multi_constructor)


def load_cfn_template(path: Path) -> dict[str, Any]:
    """Parse a CloudFormation YAML template into a plain dict."""
    text = path.read_text(encoding="utf-8")
    return yaml.load(text, Loader=_CfnLoader)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def template_text() -> str:
    assert TEMPLATE_PATH.is_file(), f"template not found at {TEMPLATE_PATH}"
    return TEMPLATE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def template(template_text: str) -> dict[str, Any]:
    parsed = yaml.load(template_text, Loader=_CfnLoader)
    assert isinstance(parsed, dict), "template did not parse into a mapping"
    return parsed


def _resources_of_type(template: dict[str, Any], type_name: str) -> dict[str, Any]:
    """Return {logical_id: resource} for every resource of the given type."""
    resources = template.get("Resources", {})
    return {
        logical_id: body
        for logical_id, body in resources.items()
        if isinstance(body, dict) and body.get("Type") == type_name
    }


def _iter_statements(policy_doc: Any) -> list[dict[str, Any]]:
    """Normalize a PolicyDocument's Statement into a list of dicts."""
    if not isinstance(policy_doc, dict):
        return []
    stmt = policy_doc.get("Statement", [])
    if isinstance(stmt, dict):
        return [stmt]
    return [s for s in stmt if isinstance(s, dict)]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


# ===========================================================================
# 1. Template validity (Req 24.3)
# ===========================================================================
def test_template_parses_into_expected_top_level_structure(
    template: dict[str, Any],
) -> None:
    """The template parses and exposes the expected top-level CFN sections."""
    assert template.get("AWSTemplateFormatVersion") == "2010-09-09"

    # Resources / Parameters / Outputs are all present and non-empty mappings.
    for section in ("Resources", "Parameters", "Outputs"):
        assert section in template, f"missing top-level section: {section}"
        assert isinstance(template[section], dict), f"{section} is not a mapping"
        assert template[section], f"{section} is empty"

    # Every resource carries a Type (basic shape sanity).
    for logical_id, body in template["Resources"].items():
        assert isinstance(body, dict), f"resource {logical_id} is not a mapping"
        assert "Type" in body, f"resource {logical_id} has no Type"


def test_cfn_lint_api_reports_no_errors(template_text: str) -> None:
    """Run cfn-lint via its Python API and assert zero error-level matches.

    Informational / warning / region-availability notes are acceptable; we gate
    only on ``error`` severity. If cfn-lint is not importable in this
    environment, skip ONLY this assertion (parse + structural + security checks
    still run) per the task's offline guarantee.
    """
    # The project's pytest config sets ``filterwarnings = ["error"]``. cfn-lint
    # emits its own DeprecationWarnings on import and during linting (third-party
    # code we do not control), which would otherwise surface as errors here. We
    # suppress those warnings locally for just this import + lint call, so a
    # genuine ImportError (cfn-lint absent) is the ONLY reason to skip.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from cfnlint import api  # type: ignore import-not-found
        except ImportError as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"cfn-lint not importable; skipping lint-API check: {exc!r}")

        # Single standard region keeps the check deterministic and offline.
        matches = api.lint(template_text, regions=["us-east-1"])

    errors = [
        m
        for m in matches
        if str(getattr(getattr(m, "rule", None), "severity", "")).lower() == "error"
    ]
    detail = "\n".join(
        f"  {m.rule.id} [{m.rule.severity}] {m.message} (line {m.linenumber})"
        for m in errors
    )
    assert not errors, f"cfn-lint reported {len(errors)} error-level match(es):\n{detail}"


# ===========================================================================
# 2a. S3 storage security constraints (Req 18.2, 20.3)
# ===========================================================================
def test_s3_bucket_is_encrypted_private_and_retained(
    template: dict[str, Any],
) -> None:
    buckets = _resources_of_type(template, "AWS::S3::Bucket")
    assert buckets, "no AWS::S3::Bucket resource found"

    for logical_id, bucket in buckets.items():
        props = bucket.get("Properties", {})

        # Server-side encryption configured (Req 18.2).
        enc = props.get("BucketEncryption")
        assert enc, f"{logical_id}: BucketEncryption not configured"
        rules = enc.get("ServerSideEncryptionConfiguration")
        assert rules, f"{logical_id}: no ServerSideEncryptionConfiguration rules"
        for rule in rules:
            sse = rule.get("ServerSideEncryptionByDefault", {})
            assert sse.get("SSEAlgorithm"), (
                f"{logical_id}: encryption rule missing SSEAlgorithm"
            )

        # Public access fully blocked: all four flags true (Req 18.2).
        pab = props.get("PublicAccessBlockConfiguration")
        assert pab, f"{logical_id}: PublicAccessBlockConfiguration not set"
        for flag in (
            "BlockPublicAcls",
            "BlockPublicPolicy",
            "IgnorePublicAcls",
            "RestrictPublicBuckets",
        ):
            assert pab.get(flag) is True, (
                f"{logical_id}: public-access flag {flag} is not true"
            )

        # Retain on delete and replace (Req 20.3).
        assert bucket.get("DeletionPolicy") == "Retain", (
            f"{logical_id}: DeletionPolicy is not Retain"
        )
        assert bucket.get("UpdateReplacePolicy") == "Retain", (
            f"{logical_id}: UpdateReplacePolicy is not Retain"
        )


# ===========================================================================
# 2b. DynamoDB retention (Req 18.4, 20.3)
# ===========================================================================
def test_dynamodb_tables_exist_and_are_retained(template: dict[str, Any]) -> None:
    tables = _resources_of_type(template, "AWS::DynamoDB::Table")
    assert tables, "no AWS::DynamoDB::Table resource found"

    for logical_id, table in tables.items():
        assert table.get("DeletionPolicy") == "Retain", (
            f"{logical_id}: DeletionPolicy is not Retain"
        )
        assert table.get("UpdateReplacePolicy") == "Retain", (
            f"{logical_id}: UpdateReplacePolicy is not Retain"
        )


# ===========================================================================
# 2c. Lambda explicit timeout + memory (Req 19.2)
# ===========================================================================
def test_every_lambda_has_explicit_timeout_and_memory(
    template: dict[str, Any],
) -> None:
    functions = _resources_of_type(template, "AWS::Lambda::Function")
    assert functions, "no AWS::Lambda::Function resource found"

    for logical_id, fn in functions.items():
        props = fn.get("Properties", {})
        assert "Timeout" in props, f"{logical_id}: no explicit Timeout"
        assert "MemorySize" in props, f"{logical_id}: no explicit MemorySize"
        # The value may be a literal or a {"Ref": <Parameter>} - both are an
        # explicit, intentional setting (not the service default).
        assert props["Timeout"] not in (None, ""), f"{logical_id}: empty Timeout"
        assert props["MemorySize"] not in (None, ""), f"{logical_id}: empty MemorySize"


# ===========================================================================
# 2d. Observability: log retention + at least one alarm (Req 19.3)
# ===========================================================================
def test_log_group_retention_and_alarm_present(template: dict[str, Any]) -> None:
    log_groups = _resources_of_type(template, "AWS::Logs::LogGroup")
    assert log_groups, "no AWS::Logs::LogGroup resource found"

    with_retention = [
        lid
        for lid, lg in log_groups.items()
        if lg.get("Properties", {}).get("RetentionInDays") not in (None, "")
    ]
    assert with_retention, "no LogGroup declares RetentionInDays"

    alarms = _resources_of_type(template, "AWS::CloudWatch::Alarm")
    assert alarms, "no AWS::CloudWatch::Alarm resource found"


# ===========================================================================
# 2e. IAM least privilege: no wildcard admin (Req 20.1)
# ===========================================================================
def test_no_iam_statement_grants_wildcard_action_and_resource(
    template: dict[str, Any],
) -> None:
    """No inline IAM role policy may allow Action "*" on Resource "*".

    Scans every AWS::IAM::Role's inline Policies. A statement is a wildcard-admin
    grant when it is Effect=Allow and lists ``*`` in BOTH Action and Resource.
    """
    roles = _resources_of_type(template, "AWS::IAM::Role")
    assert roles, "no AWS::IAM::Role resource found"

    violations: list[str] = []
    for role_id, role in roles.items():
        for policy in role.get("Properties", {}).get("Policies", []) or []:
            doc = policy.get("PolicyDocument", {}) if isinstance(policy, dict) else {}
            for i, stmt in enumerate(_iter_statements(doc)):
                if stmt.get("Effect") != "Allow":
                    continue
                actions = [str(a) for a in _as_list(stmt.get("Action"))]
                resources = [str(r) for r in _as_list(stmt.get("Resource"))]
                if "*" in actions and "*" in resources:
                    name = policy.get("PolicyName", "<unnamed>")
                    violations.append(
                        f"{role_id}/{name} statement[{i}] grants Action '*' on Resource '*'"
                    )

    assert not violations, "wildcard-admin IAM grants found:\n" + "\n".join(violations)


# ===========================================================================
# 2f. No embedded secrets (Req 20.2)
# ===========================================================================
# Heuristic secret-detection patterns. This is deliberately conservative: it
# targets shapes that are almost never legitimate in an IaC template (AWS access
# key ids, explicit secret-key assignments, private-key PEM blocks) while
# tolerating the template's benign long strings (ARNs, descriptions). Documented
# so a future reader understands what is and is not caught.
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # AWS access key id, e.g. AKIAIOSFODNN7EXAMPLE (also AGPA/ASIA/... prefixes).
    ("aws-access-key-id", re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA)[0-9A-Z]{16}\b")),
    # An explicit secret-key / password assignment with a non-trivial literal
    # value on the same line. Matches key names like aws_secret_access_key,
    # secretAccessKey, password, api_key followed by a quoted/plain value.
    (
        "secret-assignment",
        re.compile(
            r"(?i)(aws_secret_access_key|secret[_-]?access[_-]?key|"
            r"secret[_-]?key|api[_-]?key|password|passwd|private[_-]?key)"
            r"\s*[:=]\s*[\"']?[A-Za-z0-9/+=]{12,}[\"']?"
        ),
    ),
    # PEM private key block.
    ("pem-private-key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
)


def test_template_has_no_embedded_secret_literals(template_text: str) -> None:
    """Heuristic scan: no obvious secret-like literal appears in the template.

    See ``_SECRET_PATTERNS`` for exactly what is matched. Matches are reported
    with the offending substring so a real finding is actionable; the test
    fails (does not warn) on any hit, since a committed secret is a defect.
    """
    findings: list[str] = []
    for name, pattern in _SECRET_PATTERNS:
        for m in pattern.finditer(template_text):
            line_no = template_text.count("\n", 0, m.start()) + 1
            findings.append(f"[{name}] line {line_no}: {m.group(0)!r}")

    assert not findings, "possible embedded secret(s) detected:\n" + "\n".join(findings)


# ===========================================================================
# 2g. Required Outputs exist (Req 18.x / 19.1)
# ===========================================================================
def test_required_outputs_present(template: dict[str, Any]) -> None:
    """The API URL, S3 bucket name, and DynamoDB table name(s) are output.

    Outputs are matched structurally by what their ``Value`` references rather
    than by a hard-coded output key, so a future rename of an output key does
    not silently drop a required export. We require:
      * one output whose value is the HTTP API endpoint (API URL);
      * one output whose value Refs the S3 bucket (bucket name);
      * one output per DynamoDB table (each table name surfaced).
    """
    outputs = template.get("Outputs", {})
    assert outputs, "template declares no Outputs"

    bucket_ids = set(_resources_of_type(template, "AWS::S3::Bucket"))
    table_ids = set(_resources_of_type(template, "AWS::DynamoDB::Table"))
    api_ids = set(_resources_of_type(template, "AWS::ApiGatewayV2::Api"))

    def _refs(value: Any) -> set[str]:
        """Collect logical ids referenced by Ref / Fn::GetAtt in an output value."""
        found: set[str] = set()
        if isinstance(value, dict):
            for key, inner in value.items():
                if key == "Ref" and isinstance(inner, str):
                    found.add(inner)
                elif key == "Fn::GetAtt":
                    target = inner[0] if isinstance(inner, list) and inner else inner
                    if isinstance(target, str):
                        found.add(target.split(".", 1)[0])
                else:
                    found |= _refs(inner)
        elif isinstance(value, list):
            for item in value:
                found |= _refs(item)
        return found

    # Map each output to the set of resource logical ids its Value references.
    out_refs = {name: _refs(body.get("Value")) for name, body in outputs.items()}

    # API URL: an output referencing the HTTP API resource.
    assert any(out_refs[name] & api_ids for name in out_refs), (
        "no Output surfaces the API URL (reference to the ApiGatewayV2 Api)"
    )

    # S3 bucket name: an output referencing a bucket resource.
    assert any(out_refs[name] & bucket_ids for name in out_refs), (
        "no Output surfaces the S3 bucket name"
    )

    # DynamoDB table name(s): every table must be surfaced by some output.
    surfaced_tables = set().union(*out_refs.values()) if out_refs else set()
    missing_tables = table_ids - surfaced_tables
    assert not missing_tables, (
        f"DynamoDB table(s) not surfaced in Outputs: {sorted(missing_tables)}"
    )


# ===========================================================================
# Loader self-check: the CFN short-form intrinsics really did parse.
# ===========================================================================
def test_cfn_loader_materializes_short_form_intrinsics(
    template: dict[str, Any],
) -> None:
    """Sanity check that !Ref/!GetAtt/!Sub/!If became long-form mappings.

    Serializing to JSON would fail if any intrinsic had been left as an
    unresolved Python object, so this doubles as a parse-completeness guard.
    """
    # Round-trips cleanly to JSON -> every node is a plain dict/list/scalar.
    json.dumps(template)

    # The media bucket's BucketName uses !If; confirm it is a Fn::If mapping.
    bucket = _resources_of_type(template, "AWS::S3::Bucket")
    (bucket_body,) = list(bucket.values())
    name_value = bucket_body["Properties"]["BucketName"]
    assert isinstance(name_value, dict) and "Fn::If" in name_value
