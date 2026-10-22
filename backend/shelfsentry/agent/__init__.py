"""Agent loop: bounded active-perception decision loop built on Strands,
with a decision-model abstraction, deterministic fallback policy, tool-arg
validation, and a recorded decision trace.

The public decision seam is exported here: the :class:`DecisionModel` protocol
the loop depends on (design.md "Decision model abstraction", Req 7), the fixed
five-action set reused from the data contracts (``AgentAction`` /
``AGENT_ACTIONS``, Req 6.1), and the offline/causality-test
:class:`InjectedDecisionModel` (Req 7.3). The ``FallbackPolicy`` (task 6.2),
tool-arg validation (7.1), and the bounded :func:`run_agent_loop` with its
:class:`ToolClient` seam (task 7.2) are wired here. :class:`BedrockDecisionModel`
(task 7.3) is the PRIMARY path in AWS Mode (Req 7.1) and slots behind the same
``DecisionModel`` protocol as the loop's ``model`` argument - no loop change
required. It raises :class:`BedrockDecisionError` when Bedrock is unavailable or
errors, which the loop converts into a recorded fallback (``used_fallback=True``,
Req 7.2) rather than the model silently falling back and hiding the provenance.
"""

from shelfsentry.agent.bedrock import (
    DEFAULT_BEDROCK_MODEL_ID,
    BedrockDecisionError,
    BedrockDecisionModel,
)
from shelfsentry.agent.decision import (
    AGENT_ACTIONS,
    AgentAction,
    DecisionModel,
    InjectedDecisionModel,
    is_allowed_action,
)
from shelfsentry.agent.fallback import FallbackPolicy
from shelfsentry.agent.loop import ToolClient, ToolResult, run_agent_loop
from shelfsentry.agent.retry import RetryExhausted, RetryPolicy, default_sleeper
from shelfsentry.agent.tools import (
    ToolArgumentError,
    rejected_tool_execution,
    validate_tool_args,
)

__all__ = [
    "AGENT_ACTIONS",
    "DEFAULT_BEDROCK_MODEL_ID",
    "AgentAction",
    "BedrockDecisionError",
    "BedrockDecisionModel",
    "DecisionModel",
    "FallbackPolicy",
    "InjectedDecisionModel",
    "RetryExhausted",
    "RetryPolicy",
    "ToolArgumentError",
    "ToolClient",
    "ToolResult",
    "default_sleeper",
    "is_allowed_action",
    "rejected_tool_execution",
    "run_agent_loop",
    "validate_tool_args",
]
