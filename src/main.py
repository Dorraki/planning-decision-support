"""Generate draft planning decisions for Doncaster applications.

    uv run assess case-001 case-002
    PLANNING_MODEL=openai/gpt-6-luna uv run assess case-001 \
        --outputs-dir outputs/gpt-6-luna
    PLANNING_FACTS_MODEL=openai/gpt-6.1-sol uv run assess case-001 \
        --outputs-dir outputs/sol-facts

For each case this writes, under ``outputs/<case-id>/``:

- ``decision.json``: the draft decision (same schema as the expected decisions)
- ``assessment.json``: every consideration assessment, for audit
- ``report.md``: an officer-facing report with findings, policies and warnings
"""

import asyncio
import json
from pathlib import Path

import typer
from loguru import logger

from assessment import CaseAssessment, Finding, assess_case
from case_loader import PROMPT_INJECTION, CaseBundle, DocumentStatus, load_case
from evidence_check import INSUFFICIENT_EVIDENCE, evidence_alert
from policy_index import PolicyIndex

OUTPUTS_DIR = Path("outputs")
MAX_CONCURRENT_CASES = 3
LOCAL_AUTHORITY = "City of Doncaster Council"
ALERT_HEADINGS = {
    PROMPT_INJECTION: "suspected prompt injection",
    INSUFFICIENT_EVIDENCE: "evidence may be insufficient",
}

app = typer.Typer(
    name="main",
    help="Augmented planning decision-making",
    add_completion=True,
    no_args_is_help=True,
)


@app.command()
def generate_planning_assessment(
    application_names: list[str] = typer.Argument(
        ..., help="Case IDs to assess, e.g. case-001."
    ),
    refresh: bool = typer.Option(False, help="Ignore cached assessments."),
    refresh_facts: bool = typer.Option(
        False, help="Also re-extract case facts (implies --refresh)."
    ),
    outputs_dir: Path = typer.Option(
        OUTPUTS_DIR, help="Where to write <case-id>/ decision, audit and report."
    ),
) -> None:
    """Generate a draft response for one or more planning applications."""
    asyncio.run(_run(application_names, refresh, refresh_facts, outputs_dir))


async def _run(
    case_ids: list[str],
    refresh: bool,
    refresh_facts: bool,
    outputs_dir: Path,
) -> None:
    """Assess cases with bounded concurrency and write their outputs."""
    index = PolicyIndex.build()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CASES)

    async def run(case_id: str) -> None:
        async with semaphore:
            result = await assess_case(
                case_id, index, refresh=refresh, refresh_facts=refresh_facts
            )
        write_outputs(result, load_case(case_id), index, outputs_dir)

    await asyncio.gather(*(run(case_id) for case_id in case_ids))


def write_outputs(
    result: CaseAssessment,
    bundle: CaseBundle,
    index: PolicyIndex,
    outputs_dir: Path = OUTPUTS_DIR,
) -> None:
    """Write the decision, audit trail and officer report for a case.

    Args:
        result: Case assessment.
        bundle: Loaded case, for document statuses and warnings.
        index: Policy index, for policy citations.
        outputs_dir: Directory to write ``<case-id>/`` outputs under.
    """
    result.officer_alerts = bundle.officer_alerts()
    if alert := evidence_alert(bundle):
        result.officer_alerts.append(alert)
    for alert in result.officer_alerts:
        logger.warning(
            f"{result.case_id}: OFFICER ALERT - {alert.issue}"
            + (f" ({alert.document})" if alert.document else "")
        )
    case_dir = outputs_dir / result.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    decision_json = json.dumps(result.decision.model_dump(mode="json"), indent=2)
    (case_dir / "decision.json").write_text(decision_json + "\n", encoding="utf-8")
    (case_dir / "assessment.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )
    (case_dir / "report.md").write_text(
        render_report(result, bundle, index), encoding="utf-8"
    )
    logger.info(f"{result.case_id}: wrote {case_dir}")


def render_report(
    result: CaseAssessment, bundle: CaseBundle, index: PolicyIndex
) -> str:
    """Return an officer-facing markdown report for a case.

    Args:
        result: Case assessment.
        bundle: Loaded case.
        index: Policy index.
    """
    title, body = result.decision.to_markdown(
        result.case_id, local_authority=LOCAL_AUTHORITY
    ).split("\n", 1)
    lines = [title, ""]
    for alert in result.officer_alerts:
        prefix = f"Document `{alert.document}`: " if alert.document else ""
        quote = '"' if alert.issue == PROMPT_INJECTION else ""
        lines += [
            f"> **OFFICER REVIEW REQUIRED: {ALERT_HEADINGS.get(alert.issue, alert.issue)}**",
            ">",
            f"> {prefix}{alert.detail}",
            ">",
            *(f"> - {quote}{item}{quote}" for item in alert.items),
            "",
        ]
    lines += [
        body,
        "",
        "## Assessment by consideration",
        "",
        "| Consideration | Harm | Condition can resolve | Finding | Confidence "
        "| Policies |",
        "|---|---|---|---|---|---|",
    ]
    for assessment in result.assessments:
        finding = assessment.finding.value if assessment.engaged else "not engaged"
        if assessment.finding == Finding.UNACCEPTABLE and assessment.engaged:
            finding = f"**{finding}**"
        lines.append(
            f"| {assessment.consideration.value} | {assessment.harm.value} | "
            f"{'yes' if assessment.resolvable_by_condition else 'no'} | {finding} | "
            f"{assessment.confidence.value} | "
            f"{', '.join(assessment.policies_cited) or '-'} |"
        )

    notes = [a for a in result.assessments if a.engaged and a.officer_notes]
    if notes:
        lines += ["", "## Points for the officer to check", ""]
        lines += [f"- **{a.consideration.value}**: {a.officer_notes}" for a in notes]

    if bundle.warnings:
        lines += ["", "## Application pack warnings", ""]
        lines += [f"- {warning}" for warning in bundle.warnings]

    excluded = [doc for doc in bundle.documents if doc.status != DocumentStatus.CURRENT]
    if excluded:
        lines += ["", "## Documents not relied on", ""]
        lines += [f"- {doc.name}: {doc.status_reason}" for doc in excluded]

    # dict.fromkeys drops repeats but keeps the order policies were first cited.
    cited = list(
        dict.fromkeys(
            pid for a in result.assessments if a.engaged for pid in a.policies_cited
        )
    )
    if cited:
        lines += ["", "## Policies relied on", ""]
        lines += [f"- {pid}: {index.get(pid).citation}" for pid in cited]

    lines += [
        "",
        f"_Draft generated by {result.model}"
        + (f" (facts by {result.facts_model})" if result.facts_model else "")
        + " for officer review; "
        "not a planning decision._",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    app()
