"""Score draft decisions against officers' actual decisions.

    uv run evaluate                       # every case with an expected decision
    uv run evaluate case-001 --judge-model anthropic/claude-sonnet-5.5
    uv run evaluate --predictions-dir outputs/baseline

Reads ``<predictions-dir>/<case-id>/decision.json`` and
``data/decisions/<case-id>/decision.json``. Two things are scored:

- decision match: whether approve / refuse agrees with the officer;
- reasons: an LLM judge pairs each draft reason with the officer reason it
  covers and grades how closely, from "perfect" to "bad". Precision and
  recall are weighted by those grades, so a vague match earns less credit
  than a precise one.

Per-case scores and the judge's matched / missing / extra reasons are written
to ``<predictions-dir>/evaluation_<judge>.json``, so judges sit side by side.
"""

import asyncio
import json
from enum import StrEnum
from pathlib import Path

import typer
from loguru import logger
from pydantic import BaseModel, Field

from decision import Decision
from llm import get_client, model_name, model_slug

EXPECTED_DIR = Path("data") / "decisions"
OUTPUTS_DIR = Path("outputs")
SCORE_COLUMNS = ("decision_match", "precision", "recall", "f1", "mean_similarity")

JUDGE_PROMPT = """\
Compare the reasons in a draft planning decision with the officer's actual \
reasons. Pair each draft reason with the officer reason that makes the same \
planning point, if any, and grade the pair:

- perfect: same point, same substance
- excellent: same point, minor differences in wording
- very_good: same point, some detail differs
- good: same main intent, noticeably less or more specific
- okay: same topic, but misses important aspects
- bad: barely related

Each reason may appear in at most one pair. Officer reasons with no pair go \
in "missing"; draft reasons with no pair go in "extra".

Officer reasons:
{expected}

Draft reasons:
{predicted}
"""


class Similarity(StrEnum):
    """How closely a draft reason matches an officer reason."""

    PERFECT = "perfect"
    EXCELLENT = "excellent"
    VERY_GOOD = "very_good"
    GOOD = "good"
    OKAY = "okay"
    BAD = "bad"


# Credit each grade earns towards precision and recall.
SIMILARITY_WEIGHTS = {
    Similarity.PERFECT: 1.0,
    Similarity.EXCELLENT: 0.85,
    Similarity.VERY_GOOD: 0.65,
    Similarity.GOOD: 0.6,
    Similarity.OKAY: 0.2,
    Similarity.BAD: 0.0,
}


class ReasonPair(BaseModel):
    """An officer reason and the draft reason that covers it."""

    expected: str
    predicted: str
    similarity: Similarity


class Alignment(BaseModel):
    """The judge's pairing of draft reasons with officer reasons."""

    matched: list[ReasonPair] = Field(default_factory=list)
    missing: list[str] = Field(
        default_factory=list, description="Officer reasons the draft does not cover"
    )
    extra: list[str] = Field(
        default_factory=list, description="Draft reasons the officer did not give"
    )


def score_alignment(alignment: Alignment) -> dict[str, float | None]:
    """Return similarity-weighted precision, recall and F1 for an alignment.

    Args:
        alignment: The judge's pairing for one case.
    """
    weighted = sum(SIMILARITY_WEIGHTS[pair.similarity] for pair in alignment.matched)
    n_matched = len(alignment.matched)
    n_predicted = n_matched + len(alignment.extra)
    n_expected = n_matched + len(alignment.missing)
    precision = weighted / n_predicted if n_predicted else 0.0
    recall = weighted / n_expected if n_expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_similarity": round(weighted / n_matched, 2) if n_matched else None,
    }


async def judge_reasons(
    expected: list[str], predicted: list[str], judge_model: str
) -> Alignment:
    """Ask the judge model to pair draft reasons with officer reasons.

    Args:
        expected: The officer's reasons.
        predicted: The draft's reasons.
        judge_model: Model that does the pairing.

    Raises:
        ValueError: If the judge returns no parseable alignment.
    """
    if not expected or not predicted:
        return Alignment(missing=expected, extra=predicted)

    def numbered(items: list[str]) -> str:
        return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))

    response = await get_client().chat.completions.parse(
        model=judge_model,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    expected=numbered(expected), predicted=numbered(predicted)
                ),
            }
        ],
        response_format=Alignment,
    )
    alignment = response.choices[0].message.parsed
    if alignment is None:
        raise ValueError("Judge returned no alignment")
    return alignment


def load_decision(path: Path) -> Decision:
    """Load a decision JSON file.

    Args:
        path: Path to a decision JSON file.
    """
    return Decision.model_validate_json(path.read_text(encoding="utf-8"))


async def evaluate_cases(
    case_ids: list[str], judge_model: str, predictions_dir: Path = OUTPUTS_DIR
) -> dict[str, object]:
    """Score each case and return per-case results plus the mean.

    Args:
        case_ids: Cases to score.
        judge_model: Model that judges reason alignment.
        predictions_dir: Directory holding ``<case-id>/decision.json`` files.
    """
    expected = {c: load_decision(EXPECTED_DIR / c / "decision.json") for c in case_ids}
    predicted = {
        c: load_decision(predictions_dir / c / "decision.json") for c in case_ids
    }
    alignments = await asyncio.gather(
        *(
            judge_reasons(expected[c].reasons, predicted[c].reasons, judge_model)
            for c in case_ids
        )
    )

    cases: dict[str, object] = {}
    all_scores: list[dict[str, float | None]] = []
    for case_id, alignment in zip(case_ids, alignments, strict=True):
        scores: dict[str, float | None] = {
            "decision_match": float(
                expected[case_id].decision == predicted[case_id].decision
            ),
            **score_alignment(alignment),
        }
        all_scores.append(scores)
        cases[case_id] = {
            "predicted": predicted[case_id].decision,
            "expected": expected[case_id].decision,
            "scores": scores,
            "alignment": alignment.model_dump(),
        }
    return {
        "judge_model": judge_model,
        "aggregate": mean_scores(all_scores),
        "cases": cases,
    }


def mean_scores(scores: list[dict[str, float | None]]) -> dict[str, float]:
    """Return the mean of each score across cases, skipping missing values.

    Args:
        scores: Per-case scores.
    """
    means = {}
    for column in SCORE_COLUMNS:
        values = [v for s in scores if (v := s.get(column)) is not None]
        if values:
            means[column] = round(sum(values) / len(values), 4)
    return means


def format_table(report: dict[str, object]) -> str:
    """Return per-case and mean scores as a plain-text table.

    Args:
        report: Result of ``evaluate_cases``.
    """
    header = f"{'case':10s} {'pred':8s} {'real':8s} " + " ".join(
        f"{column[:14]:>14s}" for column in SCORE_COLUMNS
    )
    rows = [header, "-" * len(header)]
    cases: dict[str, dict[str, object]] = report["cases"]  # type: ignore[assignment]
    for case_id, case in cases.items():
        scores: dict[str, float] = case["scores"]  # type: ignore[assignment]
        rows.append(
            f"{case_id:10s} {case['predicted']!s:8s} {case['expected']!s:8s} "
            + " ".join(_cell(scores.get(column)) for column in SCORE_COLUMNS)
        )
    aggregate: dict[str, float] = report["aggregate"]  # type: ignore[assignment]
    rows.append("-" * len(header))
    rows.append(
        f"{'mean':28s} "
        + " ".join(_cell(aggregate.get(column)) for column in SCORE_COLUMNS)
    )
    return "\n".join(rows)


def _cell(value: float | None) -> str:
    """Format one score cell."""
    return f"{value:>14.2f}" if value is not None else f"{'-':>14s}"


app = typer.Typer(
    name="evaluate",
    help="Score draft decisions against expected decisions.",
    add_completion=False,
    no_args_is_help=False,
)


@app.command()
def main(
    case_ids: list[str] = typer.Argument(
        None, help="Cases to score. Defaults to every case with an expected decision."
    ),
    judge_model: str = typer.Option(
        None, help="Judge model. Defaults to the pipeline model."
    ),
    predictions_dir: Path = typer.Option(
        OUTPUTS_DIR, help="Directory of <case-id>/decision.json files to score."
    ),
) -> None:
    """Score draft decisions and write evaluation_<judge>.json.

    Args:
        case_ids: Cases to score.
        judge_model: Judge model; defaults to the pipeline model.
        predictions_dir: Directory of decisions to score.
    """
    case_ids = case_ids or sorted(
        path.parent.name for path in EXPECTED_DIR.glob("*/decision.json")
    )
    judge = judge_model or model_name()
    report = asyncio.run(evaluate_cases(case_ids, judge, predictions_dir))
    report_path = predictions_dir / f"evaluation_{model_slug(judge)}.json"
    report_path.write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    typer.echo(format_table(report))
    logger.info(f"Wrote {report_path}")


if __name__ == "__main__":
    app()
