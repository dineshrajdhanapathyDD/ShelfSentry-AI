"""The ``DecisionModel`` seam for the bounded active-perception agent loop.

The agent loop depends only on typed contracts and the :class:`DecisionModel`
protocol defined here, never on a concrete cloud SDK directly (design.md
"Decision model abstraction", Req 7). Three implementations slot behind the
protocol:

* :class:`BedrockDecisionModel` - the PRIMARY path in AWS Mode (Req 7.1); wired
  in a later task (7.3).
* ``FallbackPolicy`` - a thin deterministic SAFE fallback (Req 7.2, 7.4); a
  later task (6.2).
* :class:`InjectedDecisionModel` - a test/offline model whose decisions are
  supplied by the caller so causality tests run with no AWS credentials
  (Req 7.3).

This module is intentionally import-light: it pulls in only the typed contracts
and config, never ``boto3`` or ``strands`` at module top. That keeps the
offline/injected path (and the tests that use it) free of cloud dependencies.

The fixed five-action set the agent may choose from (Req 6.1) is *not*
redefined here. It is reused from :mod:`shelfsentry.data.contracts` as the
``AgentAction`` literal and the ``AGENT_ACTIONS`` tuple so there is a single
source of truth.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Protocol, runtime_checkable

from shelfsentry.config.settings import AgentConfig
from shelfsentry.data.contracts import (
    AGENT_ACTIONS,
    AgentAction,
    AgentDecision,
    TraceEntry,
    VisionFinding,
)

__all__ = [
    "AGENT_ACTIONS",
    "AgentAction",
    "DecisionModel",
    "InjectedDecisionModel",
    "is_allowed_action",
]

#: A caller-supplied decision function: ``(finding, trace, cfg) -> AgentDecision``.
#: Mirrors :meth:`DecisionModel.decide` so an injected callable is a drop-in for
#: a full model implementation.
DecisionFn = Callable[[VisionFinding, "list[TraceEntry]", AgentConfig], AgentDecision]

#: What an :class:`InjectedDecisionModel` queue entry may be: either a
#: ready-made :class:`AgentDecision` the caller fully controls, or a lightweight
#: spec mapping (at least ``action``, optionally ``rationale``/``args`` and any
#: other ``AgentDecision`` field) that this module materializes into an
#: ``AgentDecision`` with ``decided_by="injected"``.
DecisionSpec = "AgentDecision | dict[str, Any]"


def is_allowed_action(action: str) -> bool:
    """Return whether ``action`` is one of the fixed five agent actions (Req 6.1).

    The allowed set is reused verbatim from
    :data:`shelfsentry.data.contracts.AGENT_ACTIONS`; this module never defines
    its own copy.
    """
    return action in AGENT_ACTIONS


@runtime_checkable
class DecisionModel(Protocol):
    """The interface the agent loop depends on for selecting the next action.

    An implementation inspects the current :class:`VisionFinding` and the
    ordered decision ``trace`` and returns a single :class:`AgentDecision`
    whose ``action`` is one of the fixed five actions (Req 6.1). The loop
    (task 7.2) calls :meth:`decide` once per iteration.

    The concrete models slot in behind this protocol without changing the loop:
    ``BedrockDecisionModel`` (Req 7.1), ``FallbackPolicy`` (Req 7.2, 7.4), and
    :class:`InjectedDecisionModel` (Req 7.3).
    """

    def decide(
        self,
        finding: VisionFinding,
        trace: list[TraceEntry],
        cfg: AgentConfig,
    ) -> AgentDecision:
        """Select the next action and return it as an :class:`AgentDecision`."""
        ...


class InjectedDecisionModel:
    """A :class:`DecisionModel` whose decisions are supplied by the caller.

    This is the offline/causality-test driver (Req 7.3): it lets tests make
    different findings produce different actions deterministically, with no AWS
    credentials and no cloud SDK import. It supports two mutually exclusive
    modes:

    * **Queue mode** - construct with ``decisions=[...]``: a fixed, ordered
      sequence consumed one entry per :meth:`decide` call. Each entry is either
      a ready-made :class:`AgentDecision` (returned as-is, so the caller keeps
      full control of every field) or a spec mapping (e.g.
      ``{"action": "inspect_crop", "args": {...}}``) that is materialized into
      an ``AgentDecision`` with ``decided_by="injected"`` and sensible defaults
      for the required ``decision_id``/``iteration``/``rationale`` fields.
    * **Callable mode** - construct with ``fn=...``: a function
      ``(finding, trace, cfg) -> AgentDecision`` invoked per call. This is what
      causality tests use to branch on the finding's contents.

    In both modes the produced action is validated against the fixed five-action
    set (Req 6.1); an out-of-set action raises :class:`ValueError`. A caller-
    provided full :class:`AgentDecision` is already action-validated by Pydantic
    (its ``action`` field is the ``AgentAction`` literal), and is additionally
    re-checked here so a mismatch with the contract surfaces loudly.
    """

    def __init__(
        self,
        decisions: Iterable[DecisionSpec] | None = None,
        *,
        fn: DecisionFn | None = None,
    ) -> None:
        if (decisions is None) == (fn is None):
            raise ValueError(
                "InjectedDecisionModel requires exactly one of 'decisions' "
                "(a queue of decisions/specs) or 'fn' (a callable), not both "
                "or neither."
            )

        self._fn: DecisionFn | None = fn
        # Materialize the queue eagerly so construction-time errors (e.g. an
        # out-of-set action in a spec) surface immediately, and so a one-shot
        # iterable is not silently exhausted elsewhere.
        self._queue: list[DecisionSpec] | None
        self._cursor = 0
        if decisions is not None:
            self._queue = list(decisions)
        else:
            self._queue = None

    def decide(
        self,
        finding: VisionFinding,
        trace: list[TraceEntry],
        cfg: AgentConfig,
    ) -> AgentDecision:
        """Return the next caller-supplied decision (Req 7.3).

        In callable mode, the injected function is invoked with the current
        ``finding``, ``trace``, and ``cfg``. In queue mode, the next entry is
        consumed in order. The produced action is validated against the fixed
        five-action set (Req 6.1).
        """
        if self._fn is not None:
            decision = self._fn(finding, trace, cfg)
            return self._validate(decision)

        assert self._queue is not None  # one of fn/queue is always set
        if self._cursor >= len(self._queue):
            raise IndexError(
                "InjectedDecisionModel queue is exhausted: decide() was called "
                f"{self._cursor + 1} time(s) but only {len(self._queue)} "
                "decision(s) were supplied."
            )
        spec = self._queue[self._cursor]
        self._cursor += 1
        decision = self._materialize(spec, iteration=self._cursor - 1)
        return self._validate(decision)

    def _materialize(self, spec: DecisionSpec, *, iteration: int) -> AgentDecision:
        """Turn a queue entry into an :class:`AgentDecision`.

        A ready-made ``AgentDecision`` is returned unchanged (the caller owns
        every field, including ``decided_by``). A mapping spec is filled in with
        ``decided_by="injected"`` and defaults for any required field the caller
        omitted.
        """
        if isinstance(spec, AgentDecision):
            return spec
        if not isinstance(spec, dict):
            raise TypeError(
                "InjectedDecisionModel queue entries must be AgentDecision "
                f"instances or dict specs, got {type(spec).__name__}."
            )

        data: dict[str, Any] = dict(spec)
        if "action" not in data:
            raise ValueError("decision spec must include an 'action'.")
        data.setdefault("decision_id", f"injected-{iteration}")
        data.setdefault("iteration", iteration)
        data.setdefault("rationale", "injected decision")
        # Force the injected provenance for specs; a caller wanting a different
        # provenance must pass a full AgentDecision instead.
        data["decided_by"] = "injected"
        return AgentDecision.model_validate(data)

    @staticmethod
    def _validate(decision: AgentDecision) -> AgentDecision:
        """Reject a decision whose action is outside the fixed set (Req 6.1)."""
        if not isinstance(decision, AgentDecision):
            raise TypeError(
                "injected decision function must return an AgentDecision, got "
                f"{type(decision).__name__}."
            )
        if not is_allowed_action(decision.action):
            raise ValueError(
                f"injected decision action {decision.action!r} is not in the "
                f"allowed action set {AGENT_ACTIONS!r} (Req 6.1)."
            )
        return decision
