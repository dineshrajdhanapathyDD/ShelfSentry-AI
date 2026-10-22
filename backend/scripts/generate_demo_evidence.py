"""Generate real visual-to-action evidence from the ShelfSentry pipeline.

This script is the submission's **evidence generator** (task 17.3; Req 26.5). It
does NOT fabricate anything: it runs the *actual* local active-perception
pipeline (:func:`shelfsentry.pipeline.run_local_inspection` - real OpenCV 5
vision through the restricted MCP tools, the bounded agent loop, and local
on-disk persistence) on two inputs and writes out the artifacts the pipeline
really produced:

* the bundled synthetic shelf sample (:func:`shelfsentry.samples.generate_sample_shelf_png`), and
* a crafted ambiguous / conflicting multi-gap shelf scene reused from the
  reproducible labeled evaluation set (``eval-gap-two``), which the
  deterministic fallback policy is designed to escalate as conflicting evidence.

For each input it emits, under ``docs/evidence/`` (relative to the repo root):

* ``report-<name>.json``   - the canonical :class:`InspectionReport` JSON.
* ``trace-<name>.txt``     - a human-readable transcript of the decision trace
                             (each agent decision with its action + rationale,
                             and each tool execution with its tool + status),
                             emitted straight from the persisted report so it is
                             a faithful record, never hand-written.
* ``evidence-<name>-*.png``- every annotated-evidence PNG the pipeline rendered
                             and stored through the LocalStorageAdapter, fetched
                             back by its opaque media reference.

Run it from the ``backend/`` directory:

    python scripts/generate_demo_evidence.py

It prints the absolute path of every file it wrote and a short trace preview, so
the run itself is the proof that the visual-to-action loop produced the
artifacts. Re-running is safe and deterministic: the inputs are synthetic
own-work and the pipeline is a pure function of the bytes plus the deterministic
policy, so the artifacts are stable across runs (only the random inspection-id
suffix and timestamp differ).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the script runnable both as `python scripts/generate_demo_evidence.py`
# from backend/ and as a module, without requiring an editable install.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from shelfsentry.data.contracts import AgentDecision, InspectionReport, ToolExecution
from shelfsentry.data.storage import LocalStorageAdapter
from shelfsentry.eval.labeled_set import labeled_set, render_sample_png
from shelfsentry.pipeline import run_local_inspection
from shelfsentry.samples import generate_sample_shelf_png

#: Repo root is the parent of backend/.
_REPO_ROOT = _BACKEND_ROOT.parent
#: Where the committed evidence artifacts are written.
_EVIDENCE_DIR = _REPO_ROOT / "docs" / "evidence"
#: Scratch local-storage dir the pipeline persists media/reports under. It lives
#: under backend/ and is not part of the committed evidence.
_LOCAL_STORE = _BACKEND_ROOT / ".shelfsentry-evidence"


def _crafted_conflicting_sample() -> bytes:
    """Return the crafted ambiguous / conflicting shelf scene's PNG bytes.

    Reuses the reproducible labeled-set scene ``eval-gap-two`` (a stocked shelf
    with two clearly empty gaps). That scene shows clearly empty regions AND
    clearly filled regions in one frame, which is the deterministic fallback
    policy's definition of CONFLICTING evidence - the case it is designed to
    escalate to human review rather than assert an out-of-stock condition. Using
    a labeled-set scene keeps the crafted sample reproducible own-work with a
    documented expected behavior, not an invented one.
    """
    for sample in labeled_set():
        if sample.sample_id == "eval-gap-two":
            return render_sample_png(sample)
    raise RuntimeError("expected labeled sample 'eval-gap-two' to exist")


def _trace_transcript(report: InspectionReport) -> str:
    """Render a faithful text transcript of a report's decision trace.

    The transcript is derived entirely from the persisted report: it lists each
    trace entry in order - agent decisions with their chosen action, who decided,
    and the rationale; tool executions with the tool, status, and any error - so
    a reader can follow how the finding causally drove the action and the tool
    call. Nothing is invented; this is a readable projection of real data.
    """
    lines: list[str] = []
    lines.append("ShelfSentry AI - decision-trace transcript")
    lines.append("=" * 52)
    lines.append(f"inspection_id        : {report.inspection_id}")
    lines.append(f"mode                 : {report.mode}")
    lines.append(f"status               : {report.status}")
    lines.append(f"requires_human_review: {report.requires_human_review}")
    lines.append(f"initial_finding_id   : {report.initial_finding_id}")
    lines.append(f"trace entries        : {len(report.decision_trace)}")
    lines.append("")
    lines.append("Decision trace (in order):")
    lines.append("-" * 52)

    for idx, entry in enumerate(report.decision_trace):
        if isinstance(entry, AgentDecision):
            lines.append(
                f"[{idx}] AGENT DECISION  iter={entry.iteration}  "
                f"action={entry.action}  decided_by={entry.decided_by}"
                + (f"  used_fallback={entry.used_fallback}" if entry.used_fallback else "")
            )
            lines.append(f"      rationale: {entry.rationale}")
            if entry.args:
                lines.append(f"      args     : {entry.args}")
            if entry.fallback_reason:
                lines.append(f"      fallback : {entry.fallback_reason}")
        elif isinstance(entry, ToolExecution):
            lines.append(
                f"[{idx}] TOOL EXECUTION  iter={entry.iteration}  "
                f"tool={entry.tool}  status={entry.status}"
            )
            lines.append(f"      result_ref: {entry.result_ref}")
            if entry.args:
                lines.append(f"      args      : {entry.args}")
            if entry.error:
                lines.append(f"      error     : {entry.error}")
        else:  # pragma: no cover - the union has exactly two members
            lines.append(f"[{idx}] UNKNOWN ENTRY: {entry!r}")
        lines.append("")

    lines.append("Final recommendation:")
    lines.append("-" * 52)
    lines.append(report.final_recommendation)
    lines.append("")
    lines.append(f"Supporting evidence refs: {report.supporting_evidence_refs}")
    lines.append("")
    return "\n".join(lines)


def _collect_evidence_pngs(
    adapter: LocalStorageAdapter, report: InspectionReport
) -> list[tuple[str, bytes]]:
    """Fetch every stored annotated-evidence PNG for a report by its media ref.

    The pipeline renders an annotated-evidence image for the initial finding (and
    for any crop / frame probe) and stores it through the adapter, recording the
    opaque ``media://<token>`` reference. Here we read those references back
    through ``get_media`` - the same opaque-reference seam the MCP tools use - so
    the PNGs are the real rendered annotations, retrieved exactly as a client
    would retrieve them. Returns ``(finding_id, png_bytes)`` pairs.
    """
    out: list[tuple[str, bytes]] = []
    seen_refs: set[str] = set()

    # The initial finding's evidence is stored under evidence:<initial_finding_id>.
    candidate_finding_ids = [report.initial_finding_id]
    # Any tool execution that produced a finding references its evidence via the
    # result_ref (an opaque media ref for crop/frame probes).
    for entry in report.decision_trace:
        if isinstance(entry, ToolExecution) and entry.result_ref.startswith("media://"):
            if entry.result_ref not in seen_refs:
                seen_refs.add(entry.result_ref)
                try:
                    out.append((f"tool-{entry.execution_id}", adapter.get_media(entry.result_ref)))
                except Exception:  # noqa: BLE001 - best-effort evidence collection
                    pass

    # Resolve the initial finding's evidence by its inspection-scoped slot.
    from shelfsentry.data.storage import StorageMediaResolver

    resolver = StorageMediaResolver(adapter)
    for finding_id in candidate_finding_ids:
        ref = resolver.get_evidence_ref(report.inspection_id, finding_id)
        if ref and ref not in seen_refs:
            seen_refs.add(ref)
            try:
                out.append((finding_id, adapter.get_media(ref)))
            except Exception:  # noqa: BLE001
                pass
    return out


def _run_one(name: str, data: bytes, content_type: str) -> dict[str, object]:
    """Run the pipeline on one input and write its evidence artifacts.

    Returns a small summary dict describing what was written (used for the
    printed run report).
    """
    adapter = LocalStorageAdapter(str(_LOCAL_STORE))
    report = run_local_inspection(data, content_type, adapter=adapter)

    written: list[Path] = []

    # 1. Canonical report JSON.
    report_path = _EVIDENCE_DIR / f"report-{name}.json"
    report_path.write_text(report.canonical_json(), encoding="utf-8")
    written.append(report_path)

    # 2. Decision-trace transcript (emitted from the real report).
    trace_path = _EVIDENCE_DIR / f"trace-{name}.txt"
    transcript = _trace_transcript(report)
    trace_path.write_text(transcript, encoding="utf-8")
    written.append(trace_path)

    # 3. Annotated-evidence PNG(s) fetched back by opaque media reference.
    #    Filenames are stable across runs (keyed by the sample name + an ordinal,
    #    NOT the per-run random inspection id) so re-running overwrites the same
    #    committed files instead of piling up orphans.
    png_count = 0
    for ordinal, (_finding_id, png) in enumerate(_collect_evidence_pngs(adapter, report)):
        png_path = _EVIDENCE_DIR / f"evidence-{name}-{ordinal}.png"
        png_path.write_bytes(png)
        written.append(png_path)
        png_count += 1

    return {
        "name": name,
        "inspection_id": report.inspection_id,
        "status": report.status,
        "requires_human_review": report.requires_human_review,
        "trace_entries": len(report.decision_trace),
        "png_count": png_count,
        "written": [str(p) for p in written],
        "final_recommendation": report.final_recommendation,
        "transcript": transcript,
    }


def main() -> int:
    """Generate evidence for the bundled and crafted samples; print a report."""
    _EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    inputs = [
        ("bundled-sample", generate_sample_shelf_png(), "image/png"),
        ("conflicting-sample", _crafted_conflicting_sample(), "image/png"),
    ]

    summaries = [_run_one(name, data, ct) for name, data, ct in inputs]

    print("=" * 64)
    print("ShelfSentry AI - demo evidence generated")
    print(f"evidence dir: {_EVIDENCE_DIR}")
    print("=" * 64)
    for s in summaries:
        print()
        print(f"# {s['name']}")
        print(f"  inspection_id        : {s['inspection_id']}")
        print(f"  status               : {s['status']}")
        print(f"  requires_human_review: {s['requires_human_review']}")
        print(f"  trace entries        : {s['trace_entries']}")
        print(f"  annotated PNGs        : {s['png_count']}")
        print(f"  final recommendation : {s['final_recommendation']}")
        print("  files written:")
        for path in s["written"]:
            print(f"    - {path}")
        print("  trace preview:")
        preview = "\n".join(str(s["transcript"]).splitlines()[:22])
        for line in preview.splitlines():
            print(f"    | {line}")
    print()
    print("Done. All artifacts above are real pipeline output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
