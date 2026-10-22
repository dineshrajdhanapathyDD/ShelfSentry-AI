"""MCP server: a restricted set of vision tools exposed to the agent via a
maintained Python MCP SDK, reusing vision-pipeline functions.

Importing this package does NOT import the ``mcp`` SDK or start a server: the
SDK is imported lazily inside :func:`build_mcp_server`. Everything needed to
drive the four tools' logic without a live transport (the
:class:`VisionToolDispatcher`, the :class:`MediaResolver` Protocol, and the
typed argument models) is importable directly.
"""

from shelfsentry.mcp.server import (
    TOOL_ARG_MODELS,
    AnalyzeShelfImageArgs,
    CompareShelfFramesArgs,
    GetAnnotatedEvidenceArgs,
    InspectRegionCropArgs,
    McpToolError,
    MediaResolutionError,
    MediaResolver,
    VisionToolDispatcher,
    build_mcp_server,
    run_stdio_server,
    validate_tool_args,
)

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
