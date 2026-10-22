"""Restricted MCP vision-tool server (task 9.1; Req 8.1, 8.2, 8.4).

This module exposes a *minimal, security-bounded* set of exactly four tools to
the agent over a maintained Python MCP SDK (``mcp``) transport that Strands can
speak to (stdio / SSE / streamable-http). Every tool **reuses** the existing
OpenCV 5 vision functions - it never re-implements image processing or finding
assembly (Req 8.4):

======================  ====================================================================
Tool                    Delegates to
======================  ====================================================================
``analyze_shelf_image`` :func:`shelfsentry.vision.analyze.analyze_image_bytes` +
                        :func:`shelfsentry.vision.scoring.build_finding`, then
                        :func:`shelfsentry.vision.annotate.render_annotation` to store evidence
``inspect_region_crop`` :func:`shelfsentry.vision.entrypoints.inspect_region`
``compare_shelf_frames``:func:`shelfsentry.vision.entrypoints.compare_frames`
``get_annotated_evidence`` :func:`shelfsentry.vision.annotate.render_annotation` (or a stored ref)
======================  ====================================================================

Security boundary (Req 9, enforced structurally by the tool schemas): media is
**never** named by a filesystem path or URL. Each tool takes an
``inspection_id`` plus typed ids/boxes only, and the server resolves the actual
bytes through an injected :class:`MediaResolver`. The server owns the resolver;
no credential, path, URL, or shell argument is ever exposed through a tool
schema. The :class:`MediaResolver` Protocol is deliberately small and is
satisfied later by the concrete ``StorageAdapter`` (task 10.1) - this module
does **not** import or implement a concrete storage adapter, so it stays
testable with a fake resolver.

Layering. The pure, synchronous dispatch logic lives in
:class:`VisionToolDispatcher`; tests drive it directly with a fake resolver and
no live transport. :func:`build_mcp_server` wraps that dispatcher in a
``FastMCP`` server, registering one thin async tool per method so the JSON
schema is derived from the typed argument models below. The ``mcp`` SDK is
imported lazily inside :func:`build_mcp_server` so merely importing
``shelfsentry.mcp`` never hard-requires the SDK or starts a server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from shelfsentry.config.settings import VisionConfig
from shelfsentry.data.contracts import BoundingBox, VisionFinding
from shelfsentry.vision.analyze import analyze_image_bytes
from shelfsentry.vision.annotate import render_annotation
from shelfsentry.vision.entrypoints import compare_frames, inspect_region
from shelfsentry.vision.scoring import build_finding

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing the SDK at import time
    from mcp.server.fastmcp import FastMCP

__all__ = [
    "MediaResolver",
    "MediaResolutionError",
    "McpToolError",
    "AnalyzeShelfImageArgs",
    "InspectRegionCropArgs",
    "CompareShelfFramesArgs",
    "GetAnnotatedEvidenceArgs",
    "TOOL_ARG_MODELS",
    "validate_tool_args",
    "VisionToolDispatcher",
    "build_mcp_server",
    "run_stdio_server",
]


# ---------------------------------------------------------------------------
# Media resolution boundary (Req 9.1, 9.2, 9.3)
# ---------------------------------------------------------------------------


class MediaResolutionError(LookupError):
    """Raised when inspection-scoped media cannot be resolved.

    Carries no filesystem path or URL - only the inspection-scoped identifiers -
    so the security boundary is preserved even in error messages.
    """


class McpToolError(ValueError):
    """A tool invocation was rejected because its arguments are invalid (Req 8.3).

    This is the MCP-side analogue of ``shelfsentry.agent.tools.ToolArgumentError``
    and deliberately mirrors its structured, machine-readable shape (a stable
    :attr:`code`, the offending :attr:`tool`, and the offending :attr:`field`)
    so rejections look the same on both sides of the boundary. It is defined
    here rather than imported from the agent package so this module stays
    import-light and free of a circular ``mcp <-> agent`` dependency.

    Every invocation - whether it arrives over the live MCP transport or through
    a direct :class:`VisionToolDispatcher` method call - is validated against the
    tool's JSON schema (its typed ``*Args`` model) *before* any media is resolved
    or any vision function runs. When validation fails, this error is raised and
    no resolver or vision side effect occurs.

    :param code: a stable classifier, one of ``"missing_field"``,
        ``"invalid_type"``, ``"invalid_value"``, or ``"unexpected_field"``.
    :param tool: the tool whose arguments failed validation.
    :param field: the offending argument field (dotted for nested fields, e.g.
        ``"bounding_box.w"``); ``None`` when no single field is implicated.
    :param reason: a human-readable explanation of why the argument is invalid.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        tool: str,
        field: str | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.tool = tool
        self.field = field
        self.reason = reason or message

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        where = f" field={self.field!r}" if self.field is not None else ""
        return f"[{self.code}] tool={self.tool!r}{where}: {super().__str__()}"


@runtime_checkable
class MediaResolver(Protocol):
    """Inspection-scoped media resolution boundary the server depends on.

    The server addresses all media by **inspection id** (plus a typed frame or
    finding id); the resolver maps those ids to bytes and to opaque evidence
    references. This Protocol is intentionally tiny and free of any transport,
    credential, path, or URL surface, so:

    * the four tools can take only ``inspection_id`` + typed ids/boxes
      (no path/URL argument ever reaches a tool schema; Req 9.2, 9.3), and
    * the server is testable with a fake resolver (no concrete adapter here).

    The concrete ``StorageAdapter`` (task 10.1) satisfies this Protocol later
    via a thin shim; this module neither imports nor implements it.
    """

    def get_image(self, inspection_id: str) -> bytes:
        """Return the primary image bytes for ``inspection_id``.

        Raises :class:`MediaResolutionError` (or any ``LookupError``) when no
        media exists for the id.
        """
        ...

    def get_frame(self, inspection_id: str, frame_id: str) -> bytes:
        """Return the bytes of frame ``frame_id`` within ``inspection_id``.

        Raises :class:`MediaResolutionError` (or any ``LookupError``) when the
        inspection or the specific frame is unknown.
        """
        ...

    def put_evidence(self, inspection_id: str, finding_id: str, data: bytes) -> str:
        """Store annotated-evidence ``data`` and return an opaque reference.

        The returned reference is what the finding records in
        ``annotated_output_ref``; it is an opaque storage handle, never a raw
        path or URL the agent could dereference directly.
        """
        ...

    def get_evidence_ref(self, inspection_id: str, finding_id: str) -> str | None:
        """Return the stored evidence reference for a finding, or ``None``.

        ``None`` means no evidence has been stored yet for the finding, so the
        caller may render and store it on demand.
        """
        ...


# ---------------------------------------------------------------------------
# Typed tool-argument models (drive the JSON schemas; Req 8.3, 9)
# ---------------------------------------------------------------------------
#
# Each model takes ONLY an inspection_id plus typed ids/boxes. There is
# deliberately no field that names a filesystem path, a URL, a credential, or a
# shell command, so the derived JSON schema cannot expose one (Req 9.1-9.4).
# ``extra="forbid"`` rejects any unexpected argument at the boundary.


class _ToolArgs(BaseModel):
    """Base for tool argument models: strict, id-scoped, no path/URL surface."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AnalyzeShelfImageArgs(_ToolArgs):
    """Arguments for ``analyze_shelf_image``: the inspection to analyze."""

    inspection_id: str = Field(
        ...,
        min_length=1,
        description="Inspection-scoped id; the server resolves the image bytes.",
    )


class InspectRegionCropArgs(_ToolArgs):
    """Arguments for ``inspect_region_crop``: an inspection + a bounding box."""

    inspection_id: str = Field(
        ...,
        min_length=1,
        description="Inspection-scoped id; the server resolves the image bytes.",
    )
    bounding_box: BoundingBox = Field(
        ...,
        description="(x, y, w, h) integer-pixel crop box in original-image coordinates.",
    )
    region_id: str | None = Field(
        default=None,
        description="Optional id of the preceding finding's region this crop zooms into.",
    )


class CompareShelfFramesArgs(_ToolArgs):
    """Arguments for ``compare_shelf_frames``: an inspection + two frame ids."""

    inspection_id: str = Field(
        ...,
        min_length=1,
        description="Inspection-scoped id; the server resolves both frames' bytes.",
    )
    frame_id_a: str = Field(
        ..., min_length=1, description="Identifier of the earlier frame (A)."
    )
    frame_id_b: str = Field(
        ..., min_length=1, description="Identifier of the later frame (B)."
    )


class GetAnnotatedEvidenceArgs(_ToolArgs):
    """Arguments for ``get_annotated_evidence``: an inspection + a finding id."""

    inspection_id: str = Field(
        ...,
        min_length=1,
        description="Inspection-scoped id the finding belongs to.",
    )
    finding_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the finding whose Annotated_Output is requested.",
    )


# ---------------------------------------------------------------------------
# Tool -> argument-model registry and the single validation entry point
# ---------------------------------------------------------------------------
#
# The typed ``*Args`` model IS the tool's JSON-schema contract: FastMCP derives
# each tool's ``inputSchema`` from the same fields, so validating a raw argument
# mapping against the model is equivalent to validating it against the tool's
# published JSON schema. ``validate_tool_args`` is the ONE place every
# invocation - live transport or direct dispatcher call - passes through before
# execution (Req 8.3).

#: Authoritative map from tool name to its typed argument model.
TOOL_ARG_MODELS: dict[str, type[_ToolArgs]] = {
    "analyze_shelf_image": AnalyzeShelfImageArgs,
    "inspect_region_crop": InspectRegionCropArgs,
    "compare_shelf_frames": CompareShelfFramesArgs,
    "get_annotated_evidence": GetAnnotatedEvidenceArgs,
}

#: Pydantic error-type -> stable McpToolError code. Anything not listed maps to
#: ``"invalid_value"`` so an unmapped validator still yields a structured error.
_PYDANTIC_CODE_MAP: dict[str, str] = {
    "missing": "missing_field",
    "extra_forbidden": "unexpected_field",
}


def _mcp_code_for(error_type: str) -> str:
    if error_type in _PYDANTIC_CODE_MAP:
        return _PYDANTIC_CODE_MAP[error_type]
    # Pydantic groups type/parse failures under names starting with the target
    # type (e.g. ``int_parsing``, ``string_type``, ``model_type``).
    if error_type.endswith(("_type", "_parsing")):
        return "invalid_type"
    return "invalid_value"


def validate_tool_args(tool_name: str, raw_args: dict[str, Any]) -> _ToolArgs:
    """Validate a raw argument mapping against a tool's schema (Req 8.3).

    This is the single, explicit validation entry point every invocation passes
    through *before* execution. It looks up the tool's typed ``*Args`` model -
    which is the authoritative JSON-schema contract FastMCP publishes - and
    validates ``raw_args`` against it. On success it returns the validated model
    instance (with normalized, well-typed fields); on failure it raises a
    structured :class:`McpToolError` that names the offending field and reason,
    so invalid arguments are *rejected before* any media is resolved or any
    vision function runs, with no side effect.

    :param tool_name: one of the four restricted MCP tool names.
    :param raw_args: the raw, untrusted argument mapping for the invocation.
    :returns: the validated ``*Args`` instance for ``tool_name``.
    :raises McpToolError: if the tool is unknown, a required field is missing,
        a field has the wrong type or an out-of-range value, or an unexpected
        (unknown) field is present.
    """
    model = TOOL_ARG_MODELS.get(tool_name)
    if model is None:
        raise McpToolError(
            f"unknown tool {tool_name!r}; expected one of "
            f"{sorted(TOOL_ARG_MODELS)}.",
            code="invalid_value",
            tool=tool_name,
            field=None,
            reason="unknown tool",
        )
    try:
        return model.model_validate(raw_args)
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {}
        loc = first.get("loc", ())
        field = ".".join(str(p) for p in loc) if loc else None
        reason = str(first.get("msg", "invalid argument"))
        code = _mcp_code_for(str(first.get("type", "")))
        raise McpToolError(
            f"invalid arguments for {tool_name!r}: {exc.error_count()} error(s); "
            f"first: {reason}"
            + (f" at field {field!r}" if field else "")
            + ".",
            code=code,
            tool=tool_name,
            field=field,
            reason=reason,
        ) from exc


# ---------------------------------------------------------------------------
# Evidence reference helpers
# ---------------------------------------------------------------------------


def _evidence_ref(inspection_id: str, finding_id: str) -> str:
    """A deterministic, opaque default evidence reference for a finding.

    Used as the ``annotated_output_ref`` recorded on a finding when the resolver
    has not already minted one. It is an opaque handle (an ``evidence://`` URI
    over inspection-scoped ids), never a filesystem path or dereferenceable URL,
    so it does not widen the security boundary.
    """
    return f"evidence://{inspection_id}/{finding_id}"


# ---------------------------------------------------------------------------
# Pure dispatcher (testable without a live transport)
# ---------------------------------------------------------------------------


class VisionToolDispatcher:
    """Pure, synchronous dispatch for the four restricted vision tools.

    Holds the injected :class:`MediaResolver` and :class:`VisionConfig` and
    implements one method per tool. Every method resolves media by id through
    the resolver, delegates to an EXISTING vision function (no image processing
    is re-implemented here), stores/records the Annotated_Output through the
    resolver, and returns a plain JSON-able ``dict``:

    * the three analysis tools return a :class:`VisionFinding` as a dict
      (``model_dump(mode="json")``);
    * :func:`get_annotated_evidence` returns ``{"inspection_id", "finding_id",
      "annotated_output_ref"}`` - an opaque reference, not raw bytes over the
      wire by default (Req 8.2 "reference to Annotated_Output").

    Tests call these methods directly; :func:`build_mcp_server` wraps them for
    the live MCP transport.
    """

    def __init__(self, resolver: MediaResolver, cfg: VisionConfig | None = None) -> None:
        """Store the injected resolver and config.

        :param resolver: inspection-scoped media resolver (a fake in tests; the
            ``StorageAdapter`` shim in production).
        :param cfg: vision configuration; defaults to :class:`VisionConfig`
            defaults when ``None``.
        """
        self._resolver = resolver
        self._cfg = cfg or VisionConfig()

    # -- tool 1 ---------------------------------------------------------------

    def analyze_shelf_image(self, inspection_id: str) -> dict[str, Any]:
        """Analyze an inspection's image and store its Annotated_Output.

        Validates the arguments against the tool's schema *first* (Req 8.3);
        only on success does it resolve the image bytes by id, run the shared
        analysis + finding assembly (``analyze_image_bytes`` + ``build_finding``),
        render the Annotated_Output (``render_annotation``), store it through the
        resolver, and record the returned reference on the finding. Invalid
        arguments raise :class:`McpToolError` before any resolver or vision call.

        :param inspection_id: the inspection whose image to analyze.
        :returns: the assembled :class:`VisionFinding` as a JSON-able dict.
        :raises McpToolError: if the arguments fail schema validation.
        """
        args: AnalyzeShelfImageArgs = validate_tool_args(  # type: ignore[assignment]
            "analyze_shelf_image", {"inspection_id": inspection_id}
        )
        inspection_id = args.inspection_id
        data = self._resolver.get_image(inspection_id)
        finding_id = f"{inspection_id}-image"

        analysis = analyze_image_bytes(data, self._cfg)
        finding = build_finding(
            analysis,
            inspection_id=inspection_id,
            finding_id=finding_id,
            annotated_output_ref=_evidence_ref(inspection_id, finding_id),
            source="image",
            cfg=self._cfg,
        )

        annotated = render_annotation(data, finding)
        ref = self._resolver.put_evidence(inspection_id, finding_id, annotated)
        finding.annotated_output_ref = ref  # re-validated on assignment

        return finding.model_dump(mode="json")

    # -- tool 2 ---------------------------------------------------------------

    def inspect_region_crop(
        self,
        inspection_id: str,
        bounding_box: BoundingBox | dict[str, int],
        region_id: str | None = None,
    ) -> dict[str, Any]:
        """Re-analyze one crop of an inspection's image (the ``inspect_crop`` path).

        Resolves the image bytes by id and delegates to
        :func:`shelfsentry.vision.entrypoints.inspect_region` on the requested
        box, then renders and stores the crop's Annotated_Output through the
        resolver.

        :param inspection_id: the inspection whose image to crop.
        :param bounding_box: the crop box, a :class:`BoundingBox` or
            ``{x, y, w, h}`` mapping in original-image coordinates.
        :param region_id: optional id of the preceding finding's region; folded
            into the finding id so the crop is traceable to its source region.
        :returns: a :class:`VisionFinding` (``source == "crop"``) as a dict.
        :raises McpToolError: if the arguments fail schema validation (e.g. an
            empty ``inspection_id`` or a zero/negative-size box), raised before
            any resolver or vision call.
        """
        args: InspectRegionCropArgs = validate_tool_args(  # type: ignore[assignment]
            "inspect_region_crop",
            {
                "inspection_id": inspection_id,
                "bounding_box": bounding_box,
                "region_id": region_id,
            },
        )
        inspection_id = args.inspection_id
        box = args.bounding_box
        region_id = args.region_id
        data = self._resolver.get_image(inspection_id)
        suffix = region_id.strip() if isinstance(region_id, str) and region_id.strip() else "crop"
        finding_id = f"{inspection_id}-{suffix}"

        finding = inspect_region(
            data,
            box,
            self._cfg,
            inspection_id=inspection_id,
            finding_id=finding_id,
            annotated_output_ref=_evidence_ref(inspection_id, finding_id),
        )

        annotated = render_annotation(data, finding)
        ref = self._resolver.put_evidence(inspection_id, finding_id, annotated)
        finding.annotated_output_ref = ref

        return finding.model_dump(mode="json")

    # -- tool 3 ---------------------------------------------------------------

    def compare_shelf_frames(
        self, inspection_id: str, frame_id_a: str, frame_id_b: str
    ) -> dict[str, Any]:
        """Compare two frames of an inspection (the frame-diff path).

        Resolves both frames' bytes by id and delegates to
        :func:`shelfsentry.vision.entrypoints.compare_frames`, then renders and
        stores the comparison's Annotated_Output (over frame B) through the
        resolver.

        :param inspection_id: the inspection the frames belong to.
        :param frame_id_a: the earlier frame id (A).
        :param frame_id_b: the later frame id (B).
        :returns: a :class:`VisionFinding` (``source == "frame_compare"``) dict.
        :raises McpToolError: if the arguments fail schema validation, raised
            before any resolver or vision call.
        """
        args: CompareShelfFramesArgs = validate_tool_args(  # type: ignore[assignment]
            "compare_shelf_frames",
            {
                "inspection_id": inspection_id,
                "frame_id_a": frame_id_a,
                "frame_id_b": frame_id_b,
            },
        )
        inspection_id = args.inspection_id
        frame_id_a = args.frame_id_a
        frame_id_b = args.frame_id_b
        frame_a = self._resolver.get_frame(inspection_id, frame_id_a)
        frame_b = self._resolver.get_frame(inspection_id, frame_id_b)
        finding_id = f"{inspection_id}-cmp-{frame_id_a}-{frame_id_b}"

        finding = compare_frames(
            frame_a,
            frame_b,
            self._cfg,
            inspection_id=inspection_id,
            finding_id=finding_id,
            annotated_output_ref=_evidence_ref(inspection_id, finding_id),
        )

        annotated = render_annotation(frame_b, finding)
        ref = self._resolver.put_evidence(inspection_id, finding_id, annotated)
        finding.annotated_output_ref = ref

        return finding.model_dump(mode="json")

    # -- tool 4 ---------------------------------------------------------------

    def get_annotated_evidence(
        self, inspection_id: str, finding_id: str
    ) -> dict[str, Any]:
        """Return a reference to a finding's Annotated_Output (not raw bytes).

        Prefers a reference already stored by the resolver; when none exists it
        re-analyzes the inspection image, renders the Annotated_Output via
        :func:`shelfsentry.vision.annotate.render_annotation`, stores it through
        the resolver, and returns the resulting reference. Raw bytes never cross
        the wire by default (Req 8.2).

        :param inspection_id: the inspection the finding belongs to.
        :param finding_id: the finding whose Annotated_Output is requested.
        :returns: ``{"inspection_id", "finding_id", "annotated_output_ref"}``.
        :raises McpToolError: if the arguments fail schema validation, raised
            before any resolver or vision call.
        """
        args: GetAnnotatedEvidenceArgs = validate_tool_args(  # type: ignore[assignment]
            "get_annotated_evidence",
            {"inspection_id": inspection_id, "finding_id": finding_id},
        )
        inspection_id = args.inspection_id
        finding_id = args.finding_id
        existing = self._resolver.get_evidence_ref(inspection_id, finding_id)
        if existing:
            return {
                "inspection_id": inspection_id,
                "finding_id": finding_id,
                "annotated_output_ref": existing,
            }

        # Nothing stored yet: render on demand from the inspection image, reusing
        # the shared vision functions (no image processing re-implemented here).
        data = self._resolver.get_image(inspection_id)
        analysis = analyze_image_bytes(data, self._cfg)
        finding = build_finding(
            analysis,
            inspection_id=inspection_id,
            finding_id=finding_id,
            annotated_output_ref=_evidence_ref(inspection_id, finding_id),
            source="image",
            cfg=self._cfg,
        )
        annotated = render_annotation(data, finding)
        ref = self._resolver.put_evidence(inspection_id, finding_id, annotated)
        return {
            "inspection_id": inspection_id,
            "finding_id": finding_id,
            "annotated_output_ref": ref,
        }


# ---------------------------------------------------------------------------
# MCP server factory (lazy SDK import)
# ---------------------------------------------------------------------------


def build_mcp_server(
    resolver: MediaResolver,
    cfg: VisionConfig | None = None,
    *,
    name: str = "shelfsentry-vision",
) -> "FastMCP":
    """Build a ``FastMCP`` server exposing the four restricted vision tools.

    The ``mcp`` SDK is imported HERE (not at module import time) so importing
    ``shelfsentry.mcp`` never hard-requires the SDK or starts a server. Each
    tool is registered as a thin async wrapper over a
    :class:`VisionToolDispatcher` method; the tool's JSON schema is derived from
    the typed ``*Args`` models above, so no tool ever exposes a filesystem path,
    URL, credential, or shell argument (Req 9).

    :param resolver: inspection-scoped :class:`MediaResolver` (injected).
    :param cfg: vision configuration; defaults to :class:`VisionConfig`.
    :param name: server name advertised to MCP clients (e.g. Strands).
    :returns: a configured ``FastMCP`` instance with four tools registered.

    The returned server speaks any Strands-compatible transport the SDK
    supports (stdio / SSE / streamable-http) via its ``run``/``run_*_async``
    methods; :func:`run_stdio_server` is a convenience for the local default.
    """
    from mcp.server.fastmcp import FastMCP  # lazy: only when actually building a server
    from mcp.server.fastmcp.exceptions import ToolError  # lazy, same reason

    dispatcher = VisionToolDispatcher(resolver, cfg)
    server: FastMCP = FastMCP(name=name)

    def _raise_tool_error(exc: McpToolError) -> "ToolError":
        """Re-raise a structured :class:`McpToolError` as a FastMCP ``ToolError``.

        Preserves the machine-readable detail (code, tool, field, reason) in the
        message so the live transport rejects invalid arguments with the same
        structured information the direct dispatcher path raises.
        """
        return ToolError(str(exc))

    # Tools declare their fields as EXPLICIT parameters (not a single wrapper
    # model) so FastMCP derives a flat JSON schema whose top-level properties
    # are exactly inspection_id + typed ids/boxes. The typed ``*Args`` models
    # above re-validate each invocation's arguments before any vision function
    # runs (Req 8.3): a FastMCP ``ToolError`` is raised on schema-invalid input,
    # and these constructions reject anything FastMCP's looser coercion let
    # through (e.g. an empty inspection_id or a zero-size box).

    @server.tool(
        name="analyze_shelf_image",
        description=(
            "Analyze an inspection's shelf image for candidate empty-shelf "
            "regions and return a VisionFinding. Media is resolved by "
            "inspection id; no path or URL is accepted."
        ),
    )
    def analyze_shelf_image(inspection_id: str) -> dict[str, Any]:
        # The dispatcher validates against the tool schema before any vision
        # work; translate its structured rejection into a FastMCP ToolError.
        try:
            return dispatcher.analyze_shelf_image(inspection_id)
        except McpToolError as exc:
            raise _raise_tool_error(exc) from exc

    @server.tool(
        name="inspect_region_crop",
        description=(
            "Zoom in and re-analyze a single bounding-box crop of an "
            "inspection's image, returning a VisionFinding for the crop. "
            "Media is resolved by inspection id; no path or URL is accepted."
        ),
    )
    def inspect_region_crop(
        inspection_id: str,
        bounding_box: BoundingBox,
        region_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            return dispatcher.inspect_region_crop(
                inspection_id, bounding_box, region_id
            )
        except McpToolError as exc:
            raise _raise_tool_error(exc) from exc

    @server.tool(
        name="compare_shelf_frames",
        description=(
            "Compare two frames of an inspection and return a VisionFinding "
            "diff of the measured empty shelf space. Frames are resolved by "
            "inspection id and frame id; no path or URL is accepted."
        ),
    )
    def compare_shelf_frames(
        inspection_id: str, frame_id_a: str, frame_id_b: str
    ) -> dict[str, Any]:
        try:
            return dispatcher.compare_shelf_frames(
                inspection_id, frame_id_a, frame_id_b
            )
        except McpToolError as exc:
            raise _raise_tool_error(exc) from exc

    @server.tool(
        name="get_annotated_evidence",
        description=(
            "Return an opaque reference to a finding's Annotated_Output image "
            "(not raw bytes). Scoped by inspection id and finding id; no path "
            "or URL is accepted."
        ),
    )
    def get_annotated_evidence(inspection_id: str, finding_id: str) -> dict[str, Any]:
        try:
            return dispatcher.get_annotated_evidence(inspection_id, finding_id)
        except McpToolError as exc:
            raise _raise_tool_error(exc) from exc

    # Make each tool's published JSON schema an AUTHORITATIVE, CLOSED contract
    # (Req 8.3). FastMCP derives a flat schema from the explicit parameters, but
    # by default its argument model *ignores* unknown keys - so an extra field
    # would be silently dropped and the tool would run anyway. We tighten both
    # layers so unknown/extra fields are REJECTED before execution:
    #   * set ``extra="forbid"`` on each tool's argument model (what FastMCP
    #     actually validates against before calling the function), and
    #   * publish ``additionalProperties: false`` on the schema so the contract
    #     advertised to clients matches the enforced one.
    _harden_tool_schemas(server)

    return server


def _harden_tool_schemas(server: "FastMCP") -> None:
    """Close each registered tool's argument schema to unknown fields (Req 8.3).

    FastMCP's generated argument model defaults to ignoring unexpected keys,
    which would let an extra field slip through validation and the tool run with
    it silently dropped. For every registered tool we force its argument model to
    ``extra="forbid"`` (so FastMCP rejects unknown args before dispatch) and set
    ``additionalProperties: false`` on the published input schema (so the
    advertised JSON-schema contract is closed and matches enforcement). This is
    best-effort against FastMCP internals and degrades safely: the typed ``*Args``
    re-validation inside the dispatcher is the backstop either way.
    """
    manager = getattr(server, "_tool_manager", None)
    tools = getattr(manager, "_tools", None)
    if not tools:  # pragma: no cover - FastMCP internals changed; backstop still holds
        return
    for tool in tools.values():
        params = getattr(tool, "parameters", None)
        if isinstance(params, dict):
            params["additionalProperties"] = False
        meta = getattr(tool, "fn_metadata", None)
        arg_model = getattr(meta, "arg_model", None)
        if arg_model is not None:
            try:
                arg_model.model_config["extra"] = "forbid"
                arg_model.model_rebuild(force=True)
            except Exception:  # pragma: no cover - defensive; dispatcher still re-validates
                pass


def run_stdio_server(resolver: MediaResolver, cfg: VisionConfig | None = None) -> None:
    """Build and run the vision MCP server over the stdio transport.

    Convenience entry point for the local/default deployment (design.md: local
    mode runs the MCP server over stdio). Blocks until the transport closes.

    :param resolver: inspection-scoped :class:`MediaResolver` (injected).
    :param cfg: vision configuration; defaults to :class:`VisionConfig`.
    """
    server = build_mcp_server(resolver, cfg)
    server.run(transport="stdio")
