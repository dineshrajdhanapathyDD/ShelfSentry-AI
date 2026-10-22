"""MCP security-boundary and argument-rejection tests (task 9.3).

Task 9.1's ``test_server.py`` and task 9.2's ``test_arg_validation.py`` already
cover the happy path, the structured-error shape, and a curated list of invalid
inputs. This module is deliberately *complementary* and stays off that ground:

* **Security boundary (Req 9.1-9.4).** It inspects EVERY tool's *published* JSON
  ``inputSchema`` (via :meth:`FastMCP.list_tools`) and walks it RECURSIVELY -
  through ``$ref``/``$defs`` and nested object properties such as
  ``bounding_box`` - asserting no property name or declared type admits a
  filesystem path, URL/URI, raw bytes/data blob, credential/secret/token/key, or
  command/shell/exec argument. It also pins the exposed tool set to EXACTLY the
  four names (no extra tool widening capability) and asserts every object schema,
  nested ones included, carries ``additionalProperties: false`` so undeclared
  capability args cannot be smuggled in.

* **Static import guard.** A lightweight check that the server module source does
  not import ``subprocess`` / ``os.system`` / ``requests`` / ``urllib`` for tool
  execution - a robust, non-brittle guard that the tools never reach for shell
  or network capability (Req 9.3, 9.4).

* **Property 13.** A Hypothesis generator of *malformed* argument dicts drives
  both the ``validate_tool_args`` entry point and the live ``call_tool`` path,
  asserting every malformed invocation is rejected before execution with ZERO
  resolver calls (no side effect). This generalizes task 9.2's explicit cases
  rather than re-listing them.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

import shelfsentry.mcp.server as server_mod
from shelfsentry.mcp import (
    McpToolError,
    VisionToolDispatcher,
    build_mcp_server,
    validate_tool_args,
)

_EXPECTED_TOOLS = frozenset(
    {
        "analyze_shelf_image",
        "inspect_region_crop",
        "compare_shelf_frames",
        "get_annotated_evidence",
    }
)

# Substrings that, in a property NAME, would hint the tool accepts a dangerous
# capability (a path, a URL, raw bytes, a credential, or a shell command).
_FORBIDDEN_NAME_SUBSTRINGS = (
    "path",
    "url",
    "uri",
    "file",
    "dir",
    "folder",
    "cred",
    "secret",
    "token",
    "password",
    "passwd",
    "apikey",
    "access_key",
    "command",
    "cmd",
    "shell",
    "exec",
    "subprocess",
    "bytes",
    "blob",
    "payload",
    "fetch",
    "download",
    "upload",
    "endpoint",
    "host",
)

# ``key`` on its own is too broad (``finding_id`` etc. are fine), so it is only
# forbidden when it looks like a credential key. We special-case it below.
_CREDENTIAL_KEY_HINTS = ("api_key", "apikey", "access_key", "secret_key", "private_key")

# A schema-declared ``format`` that would turn a plain string into a locator or
# binary blob is itself a boundary leak, independent of the field's name.
_FORBIDDEN_FORMATS = {"uri", "iri", "uri-reference", "iri-reference", "binary", "byte", "path"}


# ---------------------------------------------------------------------------
# A resolver that records every call, so a rejection can be proven side-effect
# free: vision work is only reachable *through* the resolver.
# ---------------------------------------------------------------------------


class RecordingResolver:
    """Records every resolver method call; raises if any media is requested.

    For the boundary/rejection tests we never want a resolver call to succeed -
    if validation does its job, the resolver is never touched. Returning bytes
    is therefore unnecessary; the ``calls`` list is the whole point.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def get_image(self, inspection_id: str) -> bytes:
        self.calls.append(("get_image", inspection_id))
        raise AssertionError("resolver.get_image was reached despite invalid args")

    def get_frame(self, inspection_id: str, frame_id: str) -> bytes:
        self.calls.append(("get_frame", inspection_id, frame_id))
        raise AssertionError("resolver.get_frame was reached despite invalid args")

    def put_evidence(self, inspection_id: str, finding_id: str, data: bytes) -> str:
        self.calls.append(("put_evidence", inspection_id, finding_id))
        raise AssertionError("resolver.put_evidence was reached despite invalid args")

    def get_evidence_ref(self, inspection_id: str, finding_id: str) -> str | None:
        self.calls.append(("get_evidence_ref", inspection_id, finding_id))
        raise AssertionError("resolver.get_evidence_ref was reached despite invalid args")


@pytest.fixture()
def resolver() -> RecordingResolver:
    return RecordingResolver()


def _list_tools(server: Any) -> dict[str, Any]:
    return {t.name: t for t in asyncio.run(server.list_tools())}


def _resolve_ref(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    """Resolve a local ``$ref`` (e.g. ``#/$defs/BoundingBox``) against ``root``.

    Only local refs are expected in a pydantic/FastMCP-generated schema; a
    non-local or unresolvable ref returns the node unchanged so the walker can
    still assert on whatever is present rather than crashing.
    """
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return schema
    node: Any = root
    for part in ref.lstrip("#/").split("/"):
        if not isinstance(node, dict) or part not in node:
            return schema
        node = node[part]
    return node if isinstance(node, dict) else schema


def _iter_object_schemas(schema: dict[str, Any], root: dict[str, Any]):
    """Yield every ``type: object`` sub-schema reachable from ``schema``.

    Walks ``properties``, ``$defs``/``definitions``, and the combinator keywords
    (``allOf``/``anyOf``/``oneOf``) and array ``items``, following local
    ``$ref``s. Cycles are guarded by object identity so a self-referential
    ``$def`` cannot loop forever.
    """
    seen: set[int] = set()
    stack: list[dict[str, Any]] = [schema]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        node = _resolve_ref(node, root)
        if id(node) in seen:
            continue
        seen.add(id(node))

        if node.get("type") == "object" or "properties" in node:
            yield node

        for key in ("properties", "$defs", "definitions"):
            sub = node.get(key)
            if isinstance(sub, dict):
                stack.extend(v for v in sub.values() if isinstance(v, dict))
        for key in ("allOf", "anyOf", "oneOf"):
            sub = node.get(key)
            if isinstance(sub, list):
                stack.extend(v for v in sub if isinstance(v, dict))
        items = node.get("items")
        if isinstance(items, dict):
            stack.append(items)


def _iter_property_specs(schema: dict[str, Any], root: dict[str, Any]):
    """Yield ``(dotted_name, prop_schema)`` for every property, nested included."""
    for obj in _iter_object_schemas(schema, root):
        props = obj.get("properties")
        if not isinstance(props, dict):
            continue
        for name, spec in props.items():
            yield name, (spec if isinstance(spec, dict) else {})


def _name_is_dangerous(name: str) -> str | None:
    """Return the matched forbidden hint for ``name``, or ``None`` if it is safe."""
    lowered = name.lower()
    for hint in _CREDENTIAL_KEY_HINTS:
        if hint in lowered:
            return hint
    for bad in _FORBIDDEN_NAME_SUBSTRINGS:
        if bad in lowered:
            return bad
    # A bare, exact ``key``/``secret`` field is a credential smell; ``_id``
    # suffixes and ``finding_id`` are fine because they never *contain* "key".
    if lowered in {"key", "keys", "secret", "secrets"}:
        return lowered
    return None


# ---------------------------------------------------------------------------
# Security boundary: no tool schema (recursively) admits a dangerous capability
# ---------------------------------------------------------------------------


def test_tool_set_is_exactly_the_four_restricted_tools(
    resolver: RecordingResolver,
) -> None:
    """Exactly four tools are published; no extra tool widens the boundary (Req 9)."""
    tools = _list_tools(build_mcp_server(resolver))
    assert frozenset(tools) == _EXPECTED_TOOLS, (
        f"published tool set {sorted(tools)} is not exactly {sorted(_EXPECTED_TOOLS)}"
    )


def test_no_schema_property_admits_path_url_bytes_or_credential(
    resolver: RecordingResolver,
) -> None:
    """Recursively, no property name or type admits a dangerous capability (Req 9.1-9.4).

    Walks each tool's published JSON schema including nested objects reached via
    ``$ref`` (e.g. ``bounding_box`` -> ``BoundingBox``), asserting that:

    * no property NAME hints at a path, URL/URI, raw bytes, credential, or shell
      command, and
    * no string property declares a locator/binary ``format`` (so a plainly
      named field cannot smuggle a URL or byte blob through its format).
    """
    tools = _list_tools(build_mcp_server(resolver))
    for name, tool in tools.items():
        schema = tool.inputSchema
        specs = list(_iter_property_specs(schema, schema))
        assert specs, f"{name} schema exposed no inspectable properties"
        for prop_name, prop_schema in specs:
            matched = _name_is_dangerous(prop_name)
            assert matched is None, (
                f"{name} schema property {prop_name!r} looks like a "
                f"{matched!r} capability argument"
            )
            fmt = prop_schema.get("format")
            assert fmt not in _FORBIDDEN_FORMATS, (
                f"{name}.{prop_name} declares a boundary-leaking format {fmt!r}"
            )


def test_every_object_schema_forbids_additional_properties(
    resolver: RecordingResolver,
) -> None:
    """Every object schema - top-level and nested - sets additionalProperties:false.

    A closed schema is what stops an undeclared capability arg (a path, a URL, a
    credential) from being smuggled in alongside the declared ids/boxes. We
    assert it on the nested ``BoundingBox`` object too, not just the top level.
    """
    tools = _list_tools(build_mcp_server(resolver))
    for name, tool in tools.items():
        schema = tool.inputSchema
        object_schemas = [
            obj
            for obj in _iter_object_schemas(schema, schema)
            if obj.get("properties")
        ]
        assert object_schemas, f"{name} exposed no object schema to check"
        for obj in object_schemas:
            title = obj.get("title", "<anonymous>")
            assert obj.get("additionalProperties") is False, (
                f"{name} object schema {title!r} must set "
                f"additionalProperties:false (got {obj.get('additionalProperties')!r})"
            )


# ---------------------------------------------------------------------------
# Static guard: the server module never imports shell/network capability
# ---------------------------------------------------------------------------


def test_server_module_does_not_import_shell_or_network(
    resolver: RecordingResolver,
) -> None:
    """The server module source imports no subprocess/os.system/network client.

    A deliberately conservative, source-level guard (robust to internal
    refactors): the tools must reach neither a shell nor the network, so the
    module must not import ``subprocess``, call ``os.system``/``os.popen``, nor
    import HTTP/URL clients (``requests``, ``httpx``, ``urllib``, ``socket``).
    Comments/docstrings mentioning these words in prose are fine - we match
    import/attribute forms only.
    """
    src = inspect.getsource(server_mod)
    forbidden_fragments = (
        "import subprocess",
        "from subprocess",
        "os.system(",
        "os.popen(",
        "import requests",
        "from requests",
        "import httpx",
        "from httpx",
        "import urllib",
        "from urllib",
        "import socket",
        "from socket",
        "urlopen(",
    )
    offenders = [frag for frag in forbidden_fragments if frag in src]
    assert not offenders, (
        f"server module uses forbidden shell/network capability: {offenders}"
    )


# ---------------------------------------------------------------------------
# Property 13: Invalid tool arguments are rejected before execution
# ---------------------------------------------------------------------------
#
# Property 13: Invalid tool arguments are rejected before execution
# Validates: Requirements 6.4, 8.3
#
# For any tool invocation whose arguments fail the tool schema, the invocation
# is rejected BEFORE the tool executes and no tool side effect occurs. Rather
# than re-listing task 9.2's curated cases, we let Hypothesis synthesize a broad
# space of MALFORMED argument dicts and assert, for every one it generates, that
# both the direct ``validate_tool_args`` entry point and the live ``call_tool``
# path reject it and the resolver records ZERO calls.


# Values that are never valid for any field in any of the four tools: wrong
# scalar types, structurally broken boxes, and blank strings.
_malformed_scalars = st.one_of(
    st.integers(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.booleans(),
    st.none(),
    st.just(""),  # blank id (all id fields are min_length=1)
    st.just("   "),  # whitespace-only id -> stripped to empty
    st.lists(st.integers(), max_size=3),
    st.dictionaries(st.text(max_size=4), st.integers(), max_size=2),
)

# Bounding boxes that violate the BoundingBox contract (ge=0 on x/y, ge=1 on
# w/h) or are the wrong shape entirely.
_malformed_boxes = st.one_of(
    st.fixed_dictionaries(
        {
            "x": st.integers(max_value=-1),
            "y": st.integers(min_value=0, max_value=10),
            "w": st.integers(min_value=1, max_value=10),
            "h": st.integers(min_value=1, max_value=10),
        }
    ),
    st.fixed_dictionaries(
        {
            "x": st.integers(min_value=0, max_value=10),
            "y": st.integers(min_value=0, max_value=10),
            "w": st.integers(max_value=0),  # w < 1
            "h": st.integers(min_value=1, max_value=10),
        }
    ),
    st.fixed_dictionaries(
        {
            "x": st.integers(min_value=0, max_value=10),
            "y": st.integers(min_value=0, max_value=10),
            "w": st.integers(min_value=1, max_value=10),
            "h": st.integers(max_value=0),  # h < 1
        }
    ),
    # Missing keys / wrong element types.
    st.just({"x": 0, "y": 0, "w": 5}),
    st.just({"x": "a", "y": 0, "w": 5, "h": 5}),
    st.just("not-a-box"),
    st.integers(),
)


def _malformed_args_for(
    tool: str, include_extra_field: bool = True
) -> st.SearchStrategy[dict[str, Any]]:
    """Strategy producing argument dicts that MUST fail ``tool``'s schema.

    Each branch corrupts exactly one dimension (a bad scalar, a bad box, or -
    when ``include_extra_field`` - an unexpected extra field), keeping the
    surrounding fields well-formed so the rejection is attributable to the
    injected fault and the generator never wanders into a *valid* dict.

    ``include_extra_field`` exists because an undeclared extra key is a
    *schema*-layer fault: it is rejected by ``validate_tool_args`` and by the
    live transport, but it cannot even be expressed against the dispatcher's
    typed positional signature (which has no ``**kwargs``), so the direct-
    dispatcher property passes ``include_extra_field=False`` and the extra-field
    rejection is asserted only on the two paths that actually see the whole dict.
    """
    extra = st.dictionaries(
        st.sampled_from(["rogue", "path", "url", "cmd", "extra"]),
        st.integers(),
        min_size=1,
        max_size=2,
    )
    if tool == "analyze_shelf_image":
        branches = [st.builds(lambda v: {"inspection_id": v}, _malformed_scalars)]
        if include_extra_field:
            branches.append(st.builds(lambda e: {"inspection_id": "ok", **e}, extra))
        return st.one_of(branches)
    if tool == "inspect_region_crop":
        branches = [
            # bad id, valid box
            st.builds(
                lambda v: {"inspection_id": v, "bounding_box": {"x": 0, "y": 0, "w": 5, "h": 5}},
                _malformed_scalars,
            ),
            # valid id, bad box
            st.builds(
                lambda b: {"inspection_id": "ok", "bounding_box": b},
                _malformed_boxes,
            ),
        ]
        if include_extra_field:
            branches.append(
                st.builds(
                    lambda e: {
                        "inspection_id": "ok",
                        "bounding_box": {"x": 0, "y": 0, "w": 5, "h": 5},
                        **e,
                    },
                    extra,
                )
            )
        return st.one_of(branches)
    if tool == "compare_shelf_frames":
        branches = [
            st.builds(
                lambda v: {"inspection_id": v, "frame_id_a": "a", "frame_id_b": "b"},
                _malformed_scalars,
            ),
            st.builds(
                lambda v: {"inspection_id": "ok", "frame_id_a": v, "frame_id_b": "b"},
                _malformed_scalars,
            ),
            # missing required frame_id_b
            st.just({"inspection_id": "ok", "frame_id_a": "a"}),
        ]
        if include_extra_field:
            branches.append(
                st.builds(
                    lambda e: {"inspection_id": "ok", "frame_id_a": "a", "frame_id_b": "b", **e},
                    extra,
                )
            )
        return st.one_of(branches)
    # get_annotated_evidence
    branches = [
        st.builds(lambda v: {"inspection_id": v, "finding_id": "f"}, _malformed_scalars),
        st.builds(lambda v: {"inspection_id": "ok", "finding_id": v}, _malformed_scalars),
        st.just({"inspection_id": "ok"}),  # missing finding_id
    ]
    if include_extra_field:
        branches.append(
            st.builds(lambda e: {"inspection_id": "ok", "finding_id": "f", **e}, extra)
        )
    return st.one_of(branches)


def _tool_and_malformed_args(
    include_extra_field: bool = True,
) -> st.SearchStrategy[tuple[str, dict[str, Any]]]:
    return st.sampled_from(sorted(_EXPECTED_TOOLS)).flatmap(
        lambda tool: st.tuples(
            st.just(tool),
            _malformed_args_for(tool, include_extra_field=include_extra_field),
        )
    )


@given(data=_tool_and_malformed_args(include_extra_field=True))
def test_property13_validate_rejects_malformed_args(
    data: tuple[str, dict[str, Any]],
) -> None:
    """Property 13: ``validate_tool_args`` rejects any malformed argument dict.

    Property 13: Invalid tool arguments are rejected before execution
    Validates: Requirements 6.4, 8.3
    """
    tool, args = data
    with pytest.raises(McpToolError) as exc:
        validate_tool_args(tool, args)
    # The structured rejection names the tool and carries a stable code.
    assert exc.value.tool == tool
    assert exc.value.code in {
        "missing_field",
        "invalid_type",
        "invalid_value",
        "unexpected_field",
    }


# The dispatcher is driven through its typed positional signature, which cannot
# carry an undeclared extra key; extra-field smuggling is a schema-layer fault
# asserted on the validate + live-transport paths instead (include_extra_field
# defaults True there). Here we inject only faults the signature *can* carry - a
# bad scalar id, or a structurally invalid bounding box - so every example must
# be rejected by the dispatcher's own pre-execution validation.
@given(data=_tool_and_malformed_args(include_extra_field=False))
def test_property13_dispatcher_rejects_malformed_args_no_side_effect(
    data: tuple[str, dict[str, Any]],
) -> None:
    """Property 13: the dispatcher rejects malformed args with ZERO resolver calls.

    Property 13: Invalid tool arguments are rejected before execution
    Validates: Requirements 6.4, 8.3

    A fresh recording resolver is used per example; because every resolver
    method raises if reached, an empty ``calls`` list proves nothing ran
    downstream of validation - no media was resolved and no vision function ran.
    """
    tool, args = data
    resolver = RecordingResolver()
    dispatcher = VisionToolDispatcher(resolver)
    method = {
        "analyze_shelf_image": lambda: dispatcher.analyze_shelf_image(
            args.get("inspection_id")  # type: ignore[arg-type]
        ),
        "inspect_region_crop": lambda: dispatcher.inspect_region_crop(
            args.get("inspection_id"),  # type: ignore[arg-type]
            args.get("bounding_box"),  # type: ignore[arg-type]
            args.get("region_id"),
        ),
        "compare_shelf_frames": lambda: dispatcher.compare_shelf_frames(
            args.get("inspection_id"),  # type: ignore[arg-type]
            args.get("frame_id_a"),  # type: ignore[arg-type]
            args.get("frame_id_b"),  # type: ignore[arg-type]
        ),
        "get_annotated_evidence": lambda: dispatcher.get_annotated_evidence(
            args.get("inspection_id"),  # type: ignore[arg-type]
            args.get("finding_id"),  # type: ignore[arg-type]
        ),
    }[tool]

    with pytest.raises(McpToolError) as exc:
        method()
    assert exc.value.tool == tool  # structured rejection names the tool
    assert resolver.calls == [], (
        f"{tool} reached the resolver on malformed args {args!r}: {resolver.calls}"
    )


@given(data=_tool_and_malformed_args(include_extra_field=True))
def test_property13_live_call_tool_rejects_malformed_args_no_side_effect(
    data: tuple[str, dict[str, Any]],
) -> None:
    """Property 13: the live transport rejects malformed args with ZERO side effect.

    Property 13: Invalid tool arguments are rejected before execution
    Validates: Requirements 6.4, 8.3
    """
    tool, args = data
    resolver = RecordingResolver()
    server = build_mcp_server(resolver)
    with pytest.raises(Exception):  # FastMCP surfaces validation as a ToolError
        asyncio.run(server.call_tool(tool, args))
    assert resolver.calls == [], (
        f"{tool} reached the resolver on malformed args {args!r}: {resolver.calls}"
    )
