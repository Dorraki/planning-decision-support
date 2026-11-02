"""Single-call baseline: one model call per case, straight to a decision.

This is the comparison point for the pipeline in ``main.py``. It sends the
raw extracted text and images of every document (no cleaning, supersession,
injection screening or personal-data masking), the raw site constraints and
a list of policy titles, and asks the model for a decision directly.

    uv run baseline case-001 case-002
    uv run evaluate --predictions-dir outputs/baseline
"""

import asyncio
import json
from pathlib import Path

import typer
from loguru import logger
from openai.types.chat import ChatCompletionContentPartParam

from case_loader import APPLICATIONS_DIR, POSTCODE_RE, list_cases, read_parsed_document
from decision import Decision
from llm import get_client, image_data_url, model_name
from policy_index import NPPF_CHAPTERS, PolicyIndex, PolicySource
from site_constraints import lookup_constraints

BASELINE_DIR = Path("outputs") / "baseline"
MAX_CONCURRENT_CASES = 3

SYSTEM_PROMPT = """\
You are a Doncaster Council planning officer. Read the application pack and \
decide whether to approve the application with conditions or refuse it. Give \
the planning reasons for the decision, citing Doncaster Local Plan policies \
and NPPF paragraphs, and for an approval list the conditions with their \
reasons. Use an empty conditions list for a refusal.
"""


def load_raw_documents(case_id: str) -> list[tuple[str, str, list[bytes]]]:
    """Return ``(name, text, images)`` for every document, as parsed.

    Args:
        case_id: Case folder name, e.g. ``case-001``.
    """
    documents = []
    for path in sorted((APPLICATIONS_DIR / case_id).glob("*.json")):
        text, images, _ = read_parsed_document(path)
        documents.append((path.stem, text, images))
    return documents


def policy_titles(index: PolicyIndex) -> str:
    """Return Local Plan policy titles and NPPF chapter titles as a list.

    Args:
        index: Policy index.
    """
    local_plan = [
        f"- Policy {chunk.number}: {chunk.title}"
        for chunk in index.chunks.values()
        if chunk.source == PolicySource.LOCAL_PLAN
    ]
    nppf = [f"- Chapter {n}: {title}" for n, title in NPPF_CHAPTERS.items()]
    return (
        "Doncaster Local Plan (2021) policies:\n"
        + "\n".join(local_plan)
        + "\n\nNPPF (December 2024) chapters:\n"
        + "\n".join(nppf)
    )


def build_user_content(
    case_id: str, index: PolicyIndex
) -> list[ChatCompletionContentPartParam]:
    """Return the case pack, constraints and policy list as message content.

    Args:
        case_id: Case folder name.
        index: Policy index, for policy titles.
    """
    documents = load_raw_documents(case_id)
    postcodes = POSTCODE_RE.findall(" ".join(text for _, text, _ in documents))
    constraints = lookup_constraints(postcodes[0]) if postcodes else None
    text = "\n\n".join(
        [
            f"# Application {case_id}",
            f"Site constraints: {json.dumps(constraints)}",
            policy_titles(index),
            *(f"## Document: {name}\n{body}" for name, body, _ in documents),
        ]
    )
    content: list[ChatCompletionContentPartParam] = [{"type": "text", "text": text}]
    for name, _, images in documents:
        for image in images:
            content.append({"type": "text", "text": f"Image from document: {name}"})
            content.append(
                {"type": "image_url", "image_url": {"url": image_data_url(image)}}
            )
    return content


async def decide_case(case_id: str, index: PolicyIndex, refresh: bool) -> Decision:
    """Return the baseline decision for a case, calling the model if needed.

    Args:
        case_id: Case folder name.
        index: Policy index.
        refresh: Ignore a previously written baseline decision.

    Raises:
        ValueError: If the model returns no parseable decision.
    """
    path = BASELINE_DIR / case_id / "decision.json"
    if path.exists() and not refresh:
        return Decision.model_validate_json(path.read_text(encoding="utf-8"))

    response = await get_client().chat.completions.parse(
        model=model_name(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_content(case_id, index)},
        ],
        response_format=Decision,
    )
    decision: Decision | None = response.choices[0].message.parsed
    if decision is None:
        raise ValueError(f"No baseline decision parsed for {case_id}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(decision.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    usage = response.usage
    tokens = (
        f" ({usage.prompt_tokens} / {usage.completion_tokens} tokens)" if usage else ""
    )
    logger.info(f"{case_id}: baseline {decision.decision}{tokens}")
    return decision


async def _run(case_ids: list[str], refresh: bool) -> None:
    """Run the baseline for several cases with bounded concurrency."""
    index = PolicyIndex.build()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CASES)

    async def run(case_id: str) -> None:
        async with semaphore:
            await decide_case(case_id, index, refresh)

    await asyncio.gather(*(run(case_id) for case_id in case_ids))


app = typer.Typer(
    name="baseline",
    help="Single-call baseline decisions for comparison.",
    add_completion=False,
    no_args_is_help=False,
)


@app.command()
def main(
    case_ids: list[str] = typer.Argument(
        None, help="Case IDs, e.g. case-001. Defaults to every case."
    ),
    refresh: bool = typer.Option(False, help="Ignore previous baseline decisions."),
) -> None:
    """Write baseline decisions to outputs/baseline/<case-id>/decision.json.

    Args:
        case_ids: Cases to decide. Defaults to every case.
        refresh: Ignore previous baseline decisions.
    """
    asyncio.run(_run(case_ids or list_cases(), refresh))


if __name__ == "__main__":
    app()
