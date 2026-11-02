"""Citation-keyed index of Doncaster Local Plan policies and NPPF paragraphs.

Policies are chunked on the documents' own structure (one chunk per Local Plan
policy, one per NPPF paragraph) so they can be retrieved by the IDs planners
cite, e.g. ``LP44`` or ``NPPF135``. No embeddings are involved.

The pipeline builds the index in memory on each run (it takes under a second).
To inspect the chunks, write them to JSON after running ``uv run extract``:

    uv run policy-index
"""

import json
import re
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path

import typer
from loguru import logger
from pydantic import BaseModel, Field

# Input paths: extracted from data/policies/local_plan.pdf and nppf.pdf
EXTRACTED_DIR = Path("data") / "extracted"
LOCAL_PLAN_JSON = EXTRACTED_DIR / "policies" / "local_plan.json"
NPPF_JSON = EXTRACTED_DIR / "policies" / "nppf.json"

# Output path
DEFAULT_INDEX_PATH = EXTRACTED_DIR / "policy_index.json"


LOCAL_PLAN_POLICY_COUNT = 73  # The Local Plan has exactly 73 policies.
LOCAL_PLAN_TOC_LAST_PAGE = 11  # The Local Plan's table of contents ends on page 11.
LOCAL_PLAN_STOP_PREFIX = "Appendix 1:"
NPPF_BODY_FIRST_PAGE = 4
NPPF_STOP_PREFIX = "Annex 1:"
# Only the first 12 characters of a heading are compared with the expected title,
# because long titles wrap onto the next line in the PDF.
HEADING_MATCH_CHARS = 12

# Chapter titles are fixed by the December 2024 NPPF. Chapters and paragraphs both
# start with numbers, so matching against these titles separates "1. Introduction"
# (a chapter) from "1. The National ..." (a paragraph).
NPPF_CHAPTERS: dict[int, str] = {
    1: "Introduction",
    2: "Achieving sustainable development",
    3: "Plan-making",
    4: "Decision-making",
    5: "Delivering a sufficient supply of homes",
    6: "Building a strong, competitive economy",
    7: "Ensuring the vitality of town centres",
    8: "Promoting healthy and safe communities",
    9: "Promoting sustainable transport",
    10: "Supporting high quality communications",
    11: "Making effective use of land",
    12: "Achieving well-designed places",
    13: "Protecting Green Belt land",
    14: "Meeting the challenge of climate change, flooding and coastal change",
    15: "Conserving and enhancing the natural environment",
    16: "Conserving and enhancing the historic environment",
    17: "Facilitating the sustainable use of minerals",
}

# A contents-page line such as "Policy 44: Residential Design ..... 143";
# captures the policy number and title.
LOCAL_PLAN_TOC_RE = re.compile(r"Policy\s+(\d+):\s*(.+?)\s*\.{4,}")
# A whole line that is a Local Plan chapter heading, e.g. "Chapter 12: Design ...".
LOCAL_PLAN_CHAPTER_RE = re.compile(r"^Chapter\s+(\d+):\s*(\S.*)$")
# The repeated page header "143 [go to Contents]", which is dropped.
LOCAL_PLAN_PAGE_HEADER_RE = re.compile(r"^\d+\s+\[go to Contents\]$")
# A line that is only a number (an NPPF page number), which is dropped.
NPPF_PAGE_NUMBER_RE = re.compile(r"^\d+$")
# The "Explanation" line inside a policy's text, where the explanation is cut off.
LOCAL_PLAN_EXPLANATION_RE = re.compile(r"(?m)^Explanation\s*$")


class PolicySource(StrEnum):
    """Policy document a chunk was taken from."""

    LOCAL_PLAN = "local_plan"
    NPPF = "nppf"


class PolicyChunk(BaseModel):
    """One citable unit of policy: a Local Plan policy or an NPPF paragraph."""

    id: str = Field(..., description="Citation key, e.g. 'LP44' or 'NPPF135'")
    source: PolicySource = Field(..., description="Document the chunk came from")
    number: int = Field(..., description="Policy or paragraph number")
    title: str = Field(..., description="Policy title, or chapter title for NPPF")
    chapter: str | None = Field(default=None, description="Enclosing chapter")
    page_start: int = Field(..., description="1-based PDF page the chunk starts on")
    page_end: int = Field(..., description="1-based PDF page the chunk ends on")
    text: str = Field(..., description="Chunk text, including supporting text")

    @property  # citation for officer report
    def citation(self) -> str:
        """Return the reference as it would appear in a decision notice."""
        if self.source == PolicySource.LOCAL_PLAN:
            return f"Doncaster Local Plan Policy {self.number}: {self.title}"
        return f"NPPF paragraph {self.number} ({self.title})"

    @property  # for LLM
    def wording(self) -> str:
        """Return the policy wording without its supporting explanation.

        Local Plan chunks carry the policy box followed by an "Explanation"
        section; only the policy box is needed to apply the policy.
        """
        if self.source == PolicySource.NPPF:
            return self.text
        match = LOCAL_PLAN_EXPLANATION_RE.search(self.text)  # cut the explanation
        return self.text[: match.start()].rstrip() if match else self.text


class PolicyIndex(BaseModel):
    """Policy chunks keyed by citation ID."""

    chunks: dict[str, PolicyChunk] = Field(default_factory=dict)

    def get(self, policy_id: str) -> PolicyChunk:
        """Return the chunk for a citation ID such as 'LP44' or 'NPPF135'.

        Args:
            policy_id: Citation key, case-insensitive.

        Raises:
            KeyError: If the ID is not in the index.
        """
        return self.chunks[policy_id.upper()]

    def save(self, path: Path = DEFAULT_INDEX_PATH) -> None:
        """Write the index as JSON.

        Args:
            path: Destination file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def build(
        cls,
        local_plan_json: Path = LOCAL_PLAN_JSON,
        nppf_json: Path = NPPF_JSON,
    ) -> "PolicyIndex":
        """Build the index from extracted policy documents.

        Args:
            local_plan_json: Extracted Doncaster Local Plan JSON.
            nppf_json: Extracted NPPF JSON.

        Raises:
            FileNotFoundError: If the policy PDFs have not been parsed yet.
        """
        if not (local_plan_json.exists() and nppf_json.exists()):
            raise FileNotFoundError(
                "Extracted policy documents not found; run `uv run extract` first"
            )
        chunks = parse_local_plan(load_page_texts(local_plan_json)) + parse_nppf(
            load_page_texts(nppf_json)
        )
        return cls(chunks={chunk.id: chunk for chunk in chunks})


class _ChunkBuilder:
    """Accumulates lines for the chunk currently being parsed."""

    def __init__(
        self,
        source: PolicySource,
        number: int,
        title: str,
        chapter: str | None,
        page: int,
    ) -> None:
        self.source = source
        self.number = number
        self.title = title
        self.chapter = chapter
        self.page_start = page
        self.page_end = page
        self.lines: list[str] = []

    def add(self, page: int, line: str) -> None:
        """Append a line seen on ``page``."""
        self.lines.append(line)
        self.page_end = page

    def build(self) -> PolicyChunk:
        """Return the finished chunk."""
        prefix = "LP" if self.source == PolicySource.LOCAL_PLAN else "NPPF"
        return PolicyChunk(
            id=f"{prefix}{self.number}",
            source=self.source,
            number=self.number,
            title=self.title,
            chapter=self.chapter,
            page_start=self.page_start,
            page_end=self.page_end,
            text=_join_lines(self.lines),
        )


def load_page_texts(path: Path) -> dict[int, str]:
    """Return text per page from an extracted JSON file, skipping images.

    Args:
        path: JSON file written by ``uv run extract``.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    return {
        page["page_number"]: "\n".join(page["texts"]).replace("\r\n", "\n")
        for page in document["pages"]
    }


def parse_local_plan(pages: dict[int, str]) -> list[PolicyChunk]:
    """Split the Local Plan into one chunk per policy.

    A chunk runs from its ``Policy N: Title`` heading to the next policy,
    chapter heading or the appendices, so it carries the policy wording and its
    supporting justification. Headings must arrive in sequence and match the
    contents-page title, which rejects cross-references such as
    "(see Policy 71: iPort)".

    Args:
        pages: Page number to page text.
    """
    toc_titles = _local_plan_toc_titles(pages)
    chunks: list[PolicyChunk] = []
    current: _ChunkBuilder | None = None
    chapter: str | None = None
    next_chapter = 1
    next_policy = 1

    for page, line in _iter_lines(
        pages, LOCAL_PLAN_TOC_LAST_PAGE + 1, LOCAL_PLAN_PAGE_HEADER_RE
    ):
        if line.startswith(LOCAL_PLAN_STOP_PREFIX):
            break

        chapter_match = LOCAL_PLAN_CHAPTER_RE.match(line)
        if chapter_match and int(chapter_match.group(1)) >= next_chapter:
            chapter = line
            next_chapter = int(chapter_match.group(1)) + 1
            if current:
                chunks.append(current.build())
                current = None
            continue

        if next_policy <= LOCAL_PLAN_POLICY_COUNT and _is_local_plan_heading(
            line, next_policy, toc_titles[next_policy]
        ):
            if current:
                chunks.append(current.build())
            current = _ChunkBuilder(
                PolicySource.LOCAL_PLAN,
                next_policy,
                toc_titles[next_policy],
                chapter,
                page,
            )
            current.add(page, line)
            next_policy += 1
            continue

        if current:
            current.add(page, line)

    if current:
        chunks.append(current.build())

    if next_policy <= LOCAL_PLAN_POLICY_COUNT:
        logger.warning(f"Local Plan parsing stopped before Policy {next_policy}")
    return chunks


def parse_nppf(pages: dict[int, str]) -> list[PolicyChunk]:
    """Split the NPPF into one chunk per numbered paragraph.

    Paragraphs must arrive in sequence, so numbered list items and footnotes
    inside a paragraph are not mistaken for new paragraphs. Only the 17
    chapters are indexed; the annexes are not.

    Args:
        pages: Page number to page text.
    """
    chunks: list[PolicyChunk] = []
    current: _ChunkBuilder | None = None
    chapter: str | None = None
    next_chapter = 1
    next_paragraph = 1

    for page, line in _iter_lines(pages, NPPF_BODY_FIRST_PAGE, NPPF_PAGE_NUMBER_RE):
        if line.startswith(NPPF_STOP_PREFIX):
            break

        if _is_nppf_chapter_heading(line, next_chapter):
            chapter = f"{next_chapter}. {NPPF_CHAPTERS[next_chapter]}"
            next_chapter += 1
            if current:
                chunks.append(current.build())
                current = None
            continue

        paragraph_match = re.match(rf"^{next_paragraph}\.\s+\S", line)
        if paragraph_match:
            if current:
                chunks.append(current.build())
            title = chapter.split(". ", 1)[-1] if chapter else ""
            current = _ChunkBuilder(
                PolicySource.NPPF, next_paragraph, title, chapter, page
            )
            current.add(page, line)
            next_paragraph += 1
            continue

        if current:
            current.add(page, line)

    if current:
        chunks.append(current.build())
    return chunks


def _local_plan_toc_titles(pages: dict[int, str]) -> dict[int, str]:
    """Return policy titles from the Local Plan contents pages."""
    toc = "\n".join(
        pages[page] for page in sorted(pages) if page <= LOCAL_PLAN_TOC_LAST_PAGE
    )
    titles = {int(number): title for number, title in LOCAL_PLAN_TOC_RE.findall(toc)}
    missing = [n for n in range(1, LOCAL_PLAN_POLICY_COUNT + 1) if n not in titles]
    if missing:
        raise ValueError(f"Local Plan contents is missing policies: {missing}")
    return titles


def _is_local_plan_heading(line: str, number: int, title: str) -> bool:
    """Return whether ``line`` is the heading for Local Plan policy ``number``."""
    prefix = f"Policy {number}:"
    if not line.startswith(prefix):
        return False
    heading_title = line.removeprefix(prefix).strip()
    return heading_title[:HEADING_MATCH_CHARS] == title[:HEADING_MATCH_CHARS]


def _is_nppf_chapter_heading(line: str, number: int) -> bool:
    """Return whether ``line`` opens NPPF chapter ``number``."""
    title = NPPF_CHAPTERS.get(number)
    if title is None:
        return False
    # Long chapter titles wrap, so only the start of the title is compared.
    return line.startswith(f"{number}. {title[:HEADING_MATCH_CHARS]}")


def _iter_lines(
    pages: dict[int, str], first_page: int, furniture: re.Pattern[str]
) -> Iterator[tuple[int, str]]:
    """Yield ``(page, line)`` pairs with page headers and footers removed.

    Blank lines are yielded as empty strings so paragraph breaks survive.

    Args:
        pages: Page number to page text.
        first_page: First page to yield.
        furniture: Pattern matching whole header/footer lines to drop.
    """
    for page in sorted(pages):
        if page < first_page:
            continue
        for raw_line in pages[page].split("\n"):
            line = raw_line.strip()
            if furniture.match(line):
                continue
            yield page, line


def _join_lines(lines: list[str]) -> str:
    """Join lines, collapsing runs of blank lines into one paragraph break."""
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


app = typer.Typer(
    name="policy-index",
    help="Build the citation-keyed policy index from extracted policies.",
    add_completion=False,
    no_args_is_help=False,
)


@app.command()
def main(
    output_path: Path = typer.Option(
        DEFAULT_INDEX_PATH, help="Where to write the index JSON."
    ),
) -> None:
    """Build the policy index and write it to disk.

    Args:
        output_path: Where to write the index JSON.
    """
    index = PolicyIndex.build()
    index.save(output_path)
    counts = {
        source.value: sum(chunk.source == source for chunk in index.chunks.values())
        for source in PolicySource
    }
    logger.info(f"Wrote {len(index.chunks)} policy chunks {counts} to {output_path}")


if __name__ == "__main__":
    app()
