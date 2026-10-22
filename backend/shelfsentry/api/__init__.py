"""API layer: mode-aware request handlers that validate, route to the
pipeline, persist via a storage adapter, and return versioned responses.

Public surface (task 11.1; Req 10.4, 16.1):

* :func:`build_local_api` - factory returning the :class:`LocalApi` handler set
  wired for local mode over an injected
  :class:`~shelfsentry.data.storage.StorageAdapter`.
* :class:`LocalApi` - the handler set (create/get/list inspections, evaluation
  retrieval, annotated-evidence fetch).
* :class:`ApiResponse` / :class:`ApiError` - the versioned, mode-stamped success
  and structured-error envelopes carried on every response (Req 10.4, 10.5).
* :data:`API_VERSION` - the stable API version identifier stamped on envelopes.
"""

from shelfsentry.api.envelope import (
    API_VERSION,
    ApiError,
    ApiErrorDetail,
    ApiMode,
    ApiResponse,
    ErrorCode,
    error_from_input_validation,
    error_from_validation_error,
    not_found_error,
)
from shelfsentry.api.handlers import (
    EvalStore,
    InspectionRunner,
    LocalApi,
    build_local_api,
)

__all__ = [
    "API_VERSION",
    "ApiMode",
    "ErrorCode",
    "ApiErrorDetail",
    "ApiError",
    "ApiResponse",
    "error_from_input_validation",
    "error_from_validation_error",
    "not_found_error",
    "InspectionRunner",
    "EvalStore",
    "LocalApi",
    "build_local_api",
]
