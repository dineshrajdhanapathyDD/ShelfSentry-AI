"""AWS persistence adapter for ShelfSentry AI (task 16.1; Req 16.1, 18.3, 18.4).

:class:`AwsStorageAdapter` is the AWS-mode implementation of the single
:class:`~shelfsentry.data.storage.StorageAdapter` seam (design.md "Data Layer").
It stores media bytes in S3 and reports/jobs (plus the decision-trace/results
each report carries) as DynamoDB items. It satisfies the *same* Protocol as
:class:`~shelfsentry.data.storage.LocalStorageAdapter`, so the identical
mode-agnostic pipeline runs unchanged in either mode.

It also mints presigned S3 ``PUT`` URLs for large media uploads (Req 18.3) and
exposes :meth:`get_media_ref_for`, which lets the existing
:class:`~shelfsentry.data.storage.StorageMediaResolver` shim (task 10.1) back
the MCP server with this adapter - no AWS-specific resolver is needed.

Security boundary (Req 9, enforced structurally).
    A media *reference* is a server-minted opaque handle of the form
    ``media://<hex-token>``; the hex token is generated here, never accepted
    from the agent. The token deterministically maps to an S3 key scoped under
    the configured ``media/`` prefix (``<prefix>media/<token>.bin``). On every
    read, the token is re-validated to be pure lowercase hex of the expected
    length and the resolved key is confirmed to live under the scoped prefix,
    so a *crafted* reference (``../``, an ``s3://`` URL, an absolute path, a NUL
    byte, a foreign bucket, ...) cannot point outside the configured
    bucket/prefix - it is rejected with
    :class:`~shelfsentry.data.storage.InvalidMediaRef` before any S3 call is
    made. The manifest that maps inspection-scoped slots to refs is itself a
    DynamoDB item, so a slot lookup never dereferences a caller-supplied path.
    Record ids used as DynamoDB keys (``inspection_id``, ``job_id``) are
    sanitized the same way local ids are.

Lazy boto3 import (constraint).
    ``boto3`` is imported lazily inside methods/the constructor, so importing
    :mod:`shelfsentry.data` never requires boto3. Clients may be injected
    (``s3_client`` and ``ddb_client``) for testability; when they are not
    injected and boto3 is missing or no credentials are available, a clear
    error is raised the first time a client is actually needed - not at import.

No embedded secrets (Req 20.2).
    Bucket/table/region come from :class:`shelfsentry.config.AwsSettings` or
    explicit constructor arguments. Credentials are resolved by the standard
    boto3 provider chain at call time; nothing here reads, stores, or logs a
    credential.

Scale caveat.
    :meth:`list_reports` uses a DynamoDB ``Scan`` and filters in memory. That
    is acceptable at the small demo scale this project targets, but a Scan is
    O(table size); a production deployment with many reports should back
    History with a GSI (e.g. partitioned by ``record_type`` with a
    ``created_at`` sort key) and query it instead of scanning.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from shelfsentry.data.contracts import EvaluationResult, InspectionReport
from shelfsentry.data.storage import (
    MEDIA_REF_SCHEME,
    HistoryQuery,
    InspectionSummary,
    InvalidMediaRef,
    JobRecord,
    RecordNotFound,
    StorageError,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime boto3 import
    pass

__all__ = [
    "AwsStorageAdapter",
    "AwsClientUnavailable",
    "DEFAULT_PRESIGN_EXPIRY_SECONDS",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AwsClientUnavailable(StorageError):
    """boto3 is unavailable (not installed, or no usable client/creds).

    Raised lazily the first time an AWS client is actually needed, so importing
    :mod:`shelfsentry.data` never requires boto3 and never fails on a machine
    without AWS configured.
    """


# ---------------------------------------------------------------------------
# Constants: media token shape, key scoping, DynamoDB item shape
# ---------------------------------------------------------------------------

# A minted media token is this many random bytes rendered as lowercase hex
# (matches the local adapter's token shape so refs are indistinguishable).
_MEDIA_TOKEN_BYTES = 16
_MEDIA_TOKEN_HEX_LEN = _MEDIA_TOKEN_BYTES * 2
_HEX_TOKEN_RE = re.compile(r"\A[0-9a-f]{%d}\Z" % _MEDIA_TOKEN_HEX_LEN)

# The sub-prefix (under the configured key prefix) all media keys live under.
# Containment under this prefix is re-checked on every read (Req 9).
_MEDIA_KEY_SUBPREFIX = "media/"

# Safe record id used as a DynamoDB key: letters, digits, and a small set of
# separators only. Anything else (``/``, ``\\``, ``..``, ``:``, NUL, ...) is
# rejected so a crafted id cannot address a foreign item or escape scoping.
_SAFE_ID_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

# DynamoDB attribute names. Each item is keyed by ``pk`` and tagged by
# ``record_type`` so one table holds reports, jobs, and the media manifest.
_ATTR_PK = "pk"
_ATTR_RECORD_TYPE = "record_type"
_ATTR_DOC = "doc"  # canonical JSON of the report/job contract
_ATTR_MANIFEST = "slots"  # {slot_label: media_ref} map for a manifest item

_RT_REPORT = "report"
_RT_JOB = "job"
_RT_MANIFEST = "manifest"
_RT_EVAL = "eval"

#: Default lifetime (seconds) for a minted presigned upload URL (Req 18.3).
DEFAULT_PRESIGN_EXPIRY_SECONDS = 900  # 15 minutes


def _sanitize_record_id(raw: str, *, what: str) -> str:
    """Validate ``raw`` as a safe DynamoDB key component, or raise.

    Rejects ids that are empty, over-long, contain a path separator, a ``..``
    segment, a colon, a NUL byte, or any character outside the safe set, so a
    crafted id cannot address an item outside the intended keyspace (Req 9).
    """
    if not isinstance(raw, str) or not _SAFE_ID_RE.match(raw) or ".." in raw:
        raise InvalidMediaRef(f"unsafe {what} id")
    return raw


class AwsStorageAdapter:
    """AWS :class:`StorageAdapter`: S3 for media, DynamoDB for records.

    Construct with an :class:`shelfsentry.config.AwsSettings` *or* explicit
    ``bucket``/``table``/``region`` arguments. For tests, inject in-memory
    ``s3_client`` and ``ddb_client`` fakes implementing only the methods used
    here (S3: ``put_object``/``get_object``/``generate_presigned_url``;
    DynamoDB: ``put_item``/``get_item``/``scan``). When clients are not injected
    they are created lazily via boto3 on first use.

    ``key_prefix`` optionally namespaces every S3 object under a common prefix
    (e.g. ``"shelfsentry/"``); media keys always live under
    ``<key_prefix>media/`` and that containment is re-verified on every read.
    """

    def __init__(
        self,
        settings: Any | None = None,
        *,
        bucket: str | None = None,
        table: str | None = None,
        region: str | None = None,
        key_prefix: str = "",
        s3_client: Any | None = None,
        ddb_client: Any | None = None,
        presign_expiry_seconds: int = DEFAULT_PRESIGN_EXPIRY_SECONDS,
    ) -> None:
        # Resolve non-secret resource identifiers from settings or kwargs.
        resolved_bucket = bucket if bucket is not None else getattr(settings, "s3_bucket", None)
        resolved_table = table if table is not None else getattr(settings, "ddb_table", None)
        resolved_region = region if region is not None else getattr(settings, "region", None)

        if not resolved_bucket:
            raise StorageError("AwsStorageAdapter requires an S3 bucket name")
        if not resolved_table:
            raise StorageError("AwsStorageAdapter requires a DynamoDB table name")

        self._bucket: str = str(resolved_bucket)
        self._table: str = str(resolved_table)
        self._region: str | None = str(resolved_region) if resolved_region else None

        # Normalize the key prefix: no leading slash, exactly one trailing slash
        # when non-empty, so "<prefix>media/<token>.bin" is always well formed.
        prefix = str(key_prefix or "").strip().lstrip("/")
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        self._key_prefix: str = prefix
        self._media_prefix: str = f"{self._key_prefix}{_MEDIA_KEY_SUBPREFIX}"

        if presign_expiry_seconds <= 0:
            raise StorageError("presign_expiry_seconds must be positive")
        self._presign_expiry = int(presign_expiry_seconds)

        # Injected clients (tests) or None (lazy boto3 on first use).
        self._s3 = s3_client
        self._ddb = ddb_client

    # -- lazy client construction --------------------------------------------

    def _boto3(self) -> Any:
        """Import boto3 lazily, raising a clear error if it is unavailable."""
        try:
            import boto3  # noqa: PLC0415 - lazy by design (do not import at module load)
        except ImportError as exc:  # pragma: no cover - env without boto3
            raise AwsClientUnavailable(
                "boto3 is not installed; install it or inject s3_client/ddb_client "
                "for AWS mode."
            ) from exc
        return boto3

    def _s3_client(self) -> Any:
        """Return the S3 client, creating it lazily via boto3 if not injected."""
        if self._s3 is None:
            boto3 = self._boto3()
            try:
                self._s3 = boto3.client("s3", region_name=self._region)
            except Exception as exc:  # pragma: no cover - creds/config error path
                raise AwsClientUnavailable(
                    "could not create an S3 client (missing credentials or "
                    "configuration?)"
                ) from exc
        return self._s3

    def _ddb_client(self) -> Any:
        """Return the DynamoDB client, creating it lazily if not injected."""
        if self._ddb is None:
            boto3 = self._boto3()
            try:
                self._ddb = boto3.client("dynamodb", region_name=self._region)
            except Exception as exc:  # pragma: no cover - creds/config error path
                raise AwsClientUnavailable(
                    "could not create a DynamoDB client (missing credentials or "
                    "configuration?)"
                ) from exc
        return self._ddb

    # -- media key / ref helpers ---------------------------------------------

    def _key_for_token(self, token: str) -> str:
        """Resolve an opaque token to its scoped S3 key, safely (Req 9).

        The token must be pure lowercase hex of the expected length, and the
        resulting key is confirmed to live under the configured media prefix
        with no traversal. A crafted token therefore cannot address an object
        outside ``<key_prefix>media/`` or in another bucket.
        """
        if not isinstance(token, str) or not _HEX_TOKEN_RE.match(token):
            raise InvalidMediaRef("malformed media reference")
        key = f"{self._media_prefix}{token}.bin"
        # Defense in depth: the regex forbids separators/dots, but re-assert that
        # the composed key stays under the scoped prefix and has no traversal.
        if not key.startswith(self._media_prefix) or ".." in key:
            raise InvalidMediaRef("media reference escapes the media prefix")
        return key

    def _parse_ref(self, ref: str) -> str:
        """Extract and validate the hex token from an opaque media ``ref``.

        Rejects any ref that is not exactly ``media://<hex>`` - including an
        ``s3://bucket/key`` URL, a bare key, a foreign scheme, or a path - so a
        caller-crafted reference can never name an arbitrary S3 object (Req 9).
        """
        if not isinstance(ref, str) or not ref.startswith(MEDIA_REF_SCHEME):
            raise InvalidMediaRef("media reference has an unexpected scheme")
        return ref[len(MEDIA_REF_SCHEME) :]

    def _mint_token(self) -> str:
        """Mint a fresh random lowercase-hex media token."""
        import secrets  # noqa: PLC0415 - trivial, keep import local

        return secrets.token_hex(_MEDIA_TOKEN_BYTES)

    # -- DynamoDB item helpers ------------------------------------------------

    def _doc_pk(self, record_type: str, record_id: str) -> str:
        """Build the partition key for a report/job/manifest item."""
        return f"{record_type}#{record_id}"

    def _put_doc_item(self, record_type: str, record_id: str, doc_json: str) -> None:
        """Persist a canonical-JSON document as a DynamoDB item."""
        self._ddb_client().put_item(
            TableName=self._table,
            Item={
                _ATTR_PK: {"S": self._doc_pk(record_type, record_id)},
                _ATTR_RECORD_TYPE: {"S": record_type},
                _ATTR_DOC: {"S": doc_json},
            },
        )

    def _get_doc_item(self, record_type: str, record_id: str) -> str | None:
        """Return the stored canonical JSON for an item, or ``None``."""
        resp = self._ddb_client().get_item(
            TableName=self._table,
            Key={_ATTR_PK: {"S": self._doc_pk(record_type, record_id)}},
        )
        item = resp.get("Item") if isinstance(resp, dict) else None
        if not item:
            return None
        doc = item.get(_ATTR_DOC)
        if not isinstance(doc, dict) or "S" not in doc:
            return None
        return doc["S"]

    # -- media (S3) -----------------------------------------------------------

    def put_media(self, inspection_id: str, data: bytes, kind: str) -> str:
        """Store media bytes in S3 and return a fresh opaque reference (Req 9).

        A new random token is minted per call, the bytes are written to the
        scoped key ``<key_prefix>media/<token>.bin``, and the
        ``(inspection_id, kind)`` slot is recorded in the DynamoDB manifest item
        so the resolver shim can address media by inspection-scoped id/kind
        without the caller ever handling a key or URL.
        """
        _sanitize_record_id(inspection_id, what="inspection")
        if not isinstance(data, (bytes, bytearray)):
            raise InvalidMediaRef("media data must be bytes")

        token = self._mint_token()
        key = self._key_for_token(token)
        self._s3_client().put_object(Bucket=self._bucket, Key=key, Body=bytes(data))
        ref = f"{MEDIA_REF_SCHEME}{token}"

        # Record the slot -> ref mapping in this inspection's manifest item.
        self._set_manifest_slot(inspection_id, str(kind), ref)
        return ref

    def get_media(self, ref: str) -> bytes:
        """Return the bytes stored under an opaque ``ref`` (Req 9).

        The token is validated and scope-checked before any S3 call, so a
        crafted ref cannot read outside ``<key_prefix>media/`` or from another
        bucket.
        """
        token = self._parse_ref(ref)
        key = self._key_for_token(token)
        try:
            resp = self._s3_client().get_object(Bucket=self._bucket, Key=key)
        except Exception as exc:
            if self._is_s3_not_found(exc):
                raise RecordNotFound("media not found") from exc
            raise
        body = resp.get("Body") if isinstance(resp, dict) else None
        if body is None:
            raise RecordNotFound("media not found")
        return body.read() if hasattr(body, "read") else bytes(body)

    @staticmethod
    def _is_s3_not_found(exc: Exception) -> bool:
        """True if ``exc`` looks like an S3 missing-object error.

        Recognizes botocore ``ClientError`` NoSuchKey/404 and the in-memory
        fakes' ``KeyError``/``FileNotFoundError`` without importing botocore.
        """
        if isinstance(exc, (KeyError, FileNotFoundError)):
            return True
        resp = getattr(exc, "response", None)
        if isinstance(resp, dict):
            code = str(resp.get("Error", {}).get("Code", ""))
            status = resp.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in {"NoSuchKey", "404", "NotFound"} or status == 404:
                return True
        return False

    def mint_presigned_upload(
        self,
        inspection_id: str,
        kind: str,
        *,
        content_type: str | None = None,
        expires_in: int | None = None,
    ) -> dict[str, str]:
        """Mint a presigned S3 ``PUT`` URL for a large media upload (Req 18.3).

        A fresh opaque token/key is minted and scoped exactly like
        :meth:`put_media`, so the client uploads to a server-chosen key it can
        never influence. The returned mapping carries the opaque ``ref`` (what
        the caller later reads with :meth:`get_media`) and the presigned
        ``url``; the key is minted server-side and the upload goes only to the
        configured bucket/prefix. The slot is recorded immediately so the ref
        resolves once the client finishes uploading.
        """
        _sanitize_record_id(inspection_id, what="inspection")
        token = self._mint_token()
        key = self._key_for_token(token)

        params: dict[str, Any] = {"Bucket": self._bucket, "Key": key}
        if content_type:
            params["ContentType"] = content_type
        url = self._s3_client().generate_presigned_url(
            "put_object",
            Params=params,
            ExpiresIn=int(expires_in) if expires_in is not None else self._presign_expiry,
        )

        ref = f"{MEDIA_REF_SCHEME}{token}"
        self._set_manifest_slot(inspection_id, str(kind), ref)
        return {"ref": ref, "url": str(url)}

    def get_media_ref_for(self, inspection_id: str, kind: str) -> str | None:
        """Return the opaque ref recorded for an inspection slot, or ``None``.

        Used by :class:`~shelfsentry.data.storage.StorageMediaResolver` so the
        existing MCP resolver shim works against this adapter unchanged.
        """
        try:
            _sanitize_record_id(inspection_id, what="inspection")
        except InvalidMediaRef:
            return None
        slots = self._load_manifest(inspection_id)
        return slots.get(str(kind))

    # -- media manifest (DynamoDB item per inspection) ------------------------

    def _load_manifest(self, inspection_id: str) -> dict[str, str]:
        """Load the {slot: ref} manifest map for an inspection (may be empty)."""
        resp = self._ddb_client().get_item(
            TableName=self._table,
            Key={_ATTR_PK: {"S": self._doc_pk(_RT_MANIFEST, inspection_id)}},
        )
        item = resp.get("Item") if isinstance(resp, dict) else None
        if not item:
            return {}
        manifest = item.get(_ATTR_MANIFEST)
        if not isinstance(manifest, dict) or "M" not in manifest:
            return {}
        out: dict[str, str] = {}
        for slot, attr in manifest["M"].items():
            if isinstance(attr, dict) and "S" in attr:
                out[slot] = attr["S"]
        return out

    def _set_manifest_slot(self, inspection_id: str, slot: str, ref: str) -> None:
        """Record ``slot -> ref`` in the inspection's manifest item."""
        _sanitize_record_id(inspection_id, what="inspection")
        slots = self._load_manifest(inspection_id)
        slots[slot] = ref
        self._ddb_client().put_item(
            TableName=self._table,
            Item={
                _ATTR_PK: {"S": self._doc_pk(_RT_MANIFEST, inspection_id)},
                _ATTR_RECORD_TYPE: {"S": _RT_MANIFEST},
                _ATTR_MANIFEST: {"M": {k: {"S": v} for k, v in slots.items()}},
            },
        )

    # -- reports (DynamoDB) ---------------------------------------------------

    def save_report(self, report: InspectionReport) -> None:
        """Persist a report's canonical JSON as a DynamoDB item (Req 18.4)."""
        rid = _sanitize_record_id(report.inspection_id, what="inspection")
        self._put_doc_item(_RT_REPORT, rid, report.canonical_json())

    def get_report(self, inspection_id: str) -> InspectionReport | None:
        """Load the stored report for ``inspection_id``, or ``None``."""
        try:
            rid = _sanitize_record_id(inspection_id, what="inspection")
        except InvalidMediaRef:
            return None
        doc = self._get_doc_item(_RT_REPORT, rid)
        if doc is None:
            return None
        return InspectionReport.model_validate_json(doc)

    def list_reports(self, query: HistoryQuery) -> list[InspectionSummary]:
        """Return History summaries matching ``query`` (newest first).

        Scans the table, keeps report items, deserializes each, and applies the
        ``status`` filter, the case-insensitive ``text`` substring search over
        inspection id + final recommendation, and the ``limit`` in memory.

        Scale caveat: a ``Scan`` is O(table size). Fine for the demo scale;
        a production deployment should back History with a GSI and query it.
        """
        summaries: list[InspectionSummary] = []
        for item in self._scan_items(_RT_REPORT):
            doc = item.get(_ATTR_DOC)
            if not isinstance(doc, dict) or "S" not in doc:
                continue
            try:
                report = InspectionReport.model_validate_json(doc["S"])
            except ValueError:  # skip unreadable/foreign items
                continue
            summaries.append(InspectionSummary.from_report(report))

        if query.status is not None:
            summaries = [s for s in summaries if s.status == query.status]
        if query.text:
            needle = query.text.strip().lower()
            if needle:
                summaries = [
                    s
                    for s in summaries
                    if needle in s.inspection_id.lower()
                    or needle in s.final_recommendation.lower()
                ]

        summaries.sort(key=lambda s: (s.created_at, s.inspection_id), reverse=True)
        if query.limit is not None:
            summaries = summaries[: query.limit]
        return summaries

    def _scan_items(self, record_type: str) -> list[dict[str, Any]]:
        """Scan the table for items of ``record_type`` (paginates fully)."""
        ddb = self._ddb_client()
        items: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {
            "TableName": self._table,
            "FilterExpression": "#rt = :rt",
            "ExpressionAttributeNames": {"#rt": _ATTR_RECORD_TYPE},
            "ExpressionAttributeValues": {":rt": {"S": record_type}},
        }
        while True:
            resp = ddb.scan(**kwargs)
            items.extend(resp.get("Items", []) if isinstance(resp, dict) else [])
            start_key = resp.get("LastEvaluatedKey") if isinstance(resp, dict) else None
            if not start_key:
                break
            kwargs["ExclusiveStartKey"] = start_key
        return items

    # -- jobs (DynamoDB) ------------------------------------------------------

    def save_job(self, job: JobRecord) -> None:
        """Persist an async job record's canonical JSON as an item (Req 18.4)."""
        jid = _sanitize_record_id(job.job_id, what="job")
        self._put_doc_item(_RT_JOB, jid, job.canonical_json())

    def get_job(self, job_id: str) -> JobRecord | None:
        """Load the stored job for ``job_id``, or ``None``."""
        try:
            jid = _sanitize_record_id(job_id, what="job")
        except InvalidMediaRef:
            return None
        doc = self._get_doc_item(_RT_JOB, jid)
        if doc is None:
            return None
        return JobRecord.model_validate_json(doc)

    # -- evaluation records (DynamoDB) ----------------------------------------

    def save_eval(self, result: EvaluationResult) -> None:
        """Persist a measured ``EvaluationResult`` as an item (Req 25.4).

        Stores the record's own canonical JSON verbatim under record_type
        ``eval`` keyed by ``eval_run_id``. Nothing here derives or alters a
        metric; it is a faithful store of a measured-only record.
        """
        rid = _sanitize_record_id(result.eval_run_id, what="eval")
        self._put_doc_item(_RT_EVAL, rid, result.canonical_json())

    def get_eval(self, eval_run_id: str) -> EvaluationResult | None:
        """Load the stored evaluation record for ``eval_run_id``, or ``None``."""
        try:
            rid = _sanitize_record_id(eval_run_id, what="eval")
        except InvalidMediaRef:
            return None
        doc = self._get_doc_item(_RT_EVAL, rid)
        if doc is None:
            return None
        return EvaluationResult.model_validate_json(doc)

    def list_evals(self) -> list[EvaluationResult]:
        """Return all stored evaluation records (measured only; never fabricated).

        Scans for ``eval`` items and deserializes each. An unreadable/foreign
        item is skipped rather than invented. Newest first by ``eval_run_id``
        is not meaningful (ids are random), so results are returned in scan
        order; the UI renders each record's own measured fields.
        """
        out: list[EvaluationResult] = []
        for item in self._scan_items(_RT_EVAL):
            doc = item.get(_ATTR_DOC)
            if not isinstance(doc, dict) or "S" not in doc:
                continue
            try:
                out.append(EvaluationResult.model_validate_json(doc["S"]))
            except ValueError:
                continue
        return out
