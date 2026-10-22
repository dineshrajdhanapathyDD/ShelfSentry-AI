"""ShelfSentry AI backend package.

Agentic retail shelf inspection: OpenCV 5 vision pipeline, a bounded
Strands active-perception agent loop, a restricted MCP tool server, typed
Pydantic v2 contracts, local/AWS storage adapters, and a mode-aware API.
"""

__version__ = "0.1.0"

from shelfsentry.aws_worker import (
    AwsInspectionError,
    run_async_job,
    run_aws_inspection,
)
from shelfsentry.pipeline import (
    DispatcherToolClient,
    PipelineError,
    run_local_inspection,
)

__all__ = [
    "__version__",
    "run_local_inspection",
    "DispatcherToolClient",
    "PipelineError",
    # AWS mode (task 16.2)
    "run_aws_inspection",
    "run_async_job",
    "AwsInspectionError",
]
