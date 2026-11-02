"""Assemble a planning application pack into one structured ``CaseBundle``.

The loader reads extracted documents for a case, labels each document, tidies OCR
text, attaches site constraints from the postcode, and decides which documents
the assessment should rely on:

- personal data (emails, phone numbers, NI and bank numbers) is masked;
- superseded drawings are separated from the current material, and replies
  from the same consultee are grouped so their latest position can be read
  from content;
- documents containing suspected prompt injection are quarantined for officer
  review rather than passed to the model.

Run ``uv run extract`` first, then inspect a case with:

    uv run case-loader case-001
"""

import base64
import json
import re
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import cast

import typer
from pydantic import BaseModel, Field

from injection_screen import InjectionFinding, screen_text
from pii_screen import mask_pii
from site_constraints import lookup_constraints

APPLICATIONS_DIR = Path("data") / "extracted" / "applications"
MAX_ALERT_EXCERPTS = 3
# Source name used when citing the council's geospatial constraints lookup.
GEOSPATIAL_SOURCE = "geospatial_lookup"

# A UK postcode, e.g. "DN1 3BU" or "S70 2TA".
POSTCODE_RE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b")
# OCR renders the superscript in "m²" as "?" (e.g. "GIA 65.73m?").
SQUARE_METRES_OCR_RE = re.compile(r"(\d)\s?m\?")
DRAWING_SUPERSEDED_RE = re.compile(r"\bsuperseded\b", re.IGNORECASE)
# Splits "yorkshire_water_2" into consultee and suffix, to group replies.
REPLY_VERSION_RE = re.compile(r"^(?P<consultee>.*?)_?(?P<version>\d+)?$")
CONSULTATION_REPLY_PREFIX = "consultation_reply"

# Filenames that name the same consultee differently.
CONSULTEE_ALIASES = {"eho": "environmental_health"}


class DocumentType(StrEnum):
    """Role a document plays in the application pack."""

    APPLICATION_FORM = "application_form"
    SUPPORTING_STATEMENT = "supporting_statement"
    TECHNICAL_REPORT = "technical_report"
    CONSULTEE_REPLY = "consultee_reply"
    REPRESENTATION = "representation"
    DRAWING = "drawing"
    DRAWING_SCHEDULE = "drawing_schedule"
    OTHER = "other"


class DocumentStatus(StrEnum):
    """Whether the assessment should rely on a document."""

    CURRENT = "current"
    SUPERSEDED = "superseded"
    QUARANTINED = "quarantined"


# Order documents are presented to the model in.
DOCUMENT_TYPE_ORDER = list(DocumentType)

STATEMENT_TOKENS = {"statement", "justification", "information"}
DRAWING_TOKENS = {"plan", "plans", "elevation", "elevations", "section", "layout"}
TECHNICAL_TOKENS = {
    "assessment",
    "drainage",
    "evidence",
    "flood",
    "parking",
    "survey",
    "topographical",
    "tree",
}


class CaseDocument(BaseModel):
    """One document from an application pack."""

    name: str = Field(..., description="Filename stem of the source PDF")
    doc_type: DocumentType
    status: DocumentStatus = DocumentStatus.CURRENT
    status_reason: str | None = Field(
        default=None, description="Why the document is not current"
    )
    reply_group: str | None = Field(
        default=None,
        description="Consultee shared with other replies whose order is unknown",
    )
    superseded_by: str | None = Field(
        default=None, description="Name of the document that replaces this one"
    )
    page_count: int
    text: str
    images: list[bytes] = Field(default_factory=list, exclude=True, repr=False)
    injection_findings: list[InjectionFinding] = Field(default_factory=list)
    pii_masked: dict[str, int] = Field(
        default_factory=dict, description="Personal data masked, by kind"
    )


PROMPT_INJECTION = "suspected_prompt_injection"


class OfficerAlert(BaseModel):
    """Something an officer must review before relying on the draft."""

    issue: str = Field(..., description="e.g. 'suspected_prompt_injection'")
    document: str | None = Field(
        default=None, description="Document concerned, if the alert is about one"
    )
    detail: str
    items: list[str] = Field(
        default_factory=list, description="Excerpts, or evidence to request"
    )


class CaseBundle(BaseModel):
    """Everything known about one application before any model call."""

    case_id: str
    postcode: str | None
    constraints: dict[str, object] | None = Field(
        default=None, description="Planning constraints from the postcode lookup"
    )
    documents: list[CaseDocument]
    warnings: list[str] = Field(default_factory=list)

    def documents_with_status(self, status: DocumentStatus) -> list[CaseDocument]:
        """Return documents with the given status, in presentation order.

        Args:
            status: Status to filter on.
        """
        return [doc for doc in self.documents if doc.status == status]

    def officer_alerts(self) -> list[OfficerAlert]:
        """Return alerts for documents quarantined as suspected prompt injection.

        The case is still assessed without them; the alert tells the officer
        why a document was withheld and what it tried to do.
        """
        return [
            OfficerAlert(
                document=doc.name,
                issue=PROMPT_INJECTION,
                detail=(
                    "Contains text that tries to direct the assessment ("
                    + ", ".join(sorted({f.pattern for f in doc.injection_findings}))
                    + "). It was quarantined: its content was not sent to the model "
                    "and did not influence this draft. Review it before relying on "
                    "this draft."
                ),
                items=[f.excerpt for f in doc.injection_findings[:MAX_ALERT_EXCERPTS]],
            )
            for doc in self.documents_with_status(DocumentStatus.QUARANTINED)
        ]

    def current_images(self) -> list[tuple[str, bytes]]:
        """Return ``(document name, image)`` pairs from current documents.

        Covers drawings and scanned documents whose content is only in images.
        """
        return [
            (doc.name, image)
            for doc in self.documents_with_status(DocumentStatus.CURRENT)
            for image in doc.images
        ]

    def render_text(self) -> str:
        """Render the bundle as prompt-ready text.

        Current documents are included in full; replies from the same
        consultee carry a shared ``reply_group``. Superseded drawings and
        quarantined documents are listed by name only.
        """
        constraint_lines = describe_constraints(self.constraints)
        sections = [
            f"# Case {self.case_id}",
            f"Site postcode: {self.postcode or 'not found'}",
            f"## Site constraints (source: {GEOSPATIAL_SOURCE})",
            "\n".join(f"- {line}" for line in constraint_lines),
            "## Current documents",
        ]
        sections += [
            _render_document(doc)
            for doc in self.documents_with_status(DocumentStatus.CURRENT)
        ]

        excluded = self.documents_with_status(DocumentStatus.SUPERSEDED)
        excluded += self.documents_with_status(DocumentStatus.QUARANTINED)
        if excluded:
            sections.append("## Excluded documents (not to be relied on)")
            sections += [f"- {doc.name}: {doc.status_reason}" for doc in excluded]

        return "\n\n".join(sections)


def list_cases(applications_dir: Path = APPLICATIONS_DIR) -> list[str]:
    """Return the case IDs available in the extracted applications directory.

    Args:
        applications_dir: Directory containing one folder per case.
    """
    return sorted(path.name for path in applications_dir.iterdir() if path.is_dir())


def load_case(case_id: str, applications_dir: Path = APPLICATIONS_DIR) -> CaseBundle:
    """Load, label and screen every document for a case.

    Args:
        case_id: Case folder name, e.g. ``case-003``.
        applications_dir: Directory containing one folder per case.

    Raises:
        FileNotFoundError: If the case has no extracted documents.
    """
    case_dir = applications_dir / case_id
    paths = sorted(case_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(
            f"No extracted documents in {case_dir}; run `uv run extract` first"
        )

    documents = [_load_document(path) for path in paths]
    warnings: list[str] = []
    for doc in documents:
        if not doc.text.strip():
            warnings.append(
                f"{doc.name}: no extractable text (likely scanned); "
                f"{len(doc.images)} page image(s) need reading"
            )
        if doc.doc_type == DocumentType.OTHER:
            warnings.append(f"{doc.name}: unrecognised document type")
        if doc.pii_masked:
            masked = ", ".join(f"{kind} x{n}" for kind, n in doc.pii_masked.items())
            warnings.append(f"{doc.name}: personal data masked ({masked})")

    _mark_superseded_drawings(documents)
    warnings += _group_consultee_replies(documents)
    warnings += _quarantine_injections(documents)

    postcode, postcode_warnings = _find_postcode(documents)
    warnings += postcode_warnings
    constraints = lookup_constraints(postcode) if postcode else None
    if postcode and constraints is None:
        warnings.append(f"Postcode {postcode} not found in site constraints")

    # The model sees documents in a fixed order, so the output is the same each run.
    documents.sort(key=lambda doc: (DOCUMENT_TYPE_ORDER.index(doc.doc_type), doc.name))
    return CaseBundle(
        case_id=case_id,
        postcode=postcode,
        constraints=constraints,
        documents=documents,
        warnings=warnings,
    )


def describe_constraints(constraints: dict[str, object] | None) -> list[str]:
    """Describe geospatial constraints in plain planning terms.

    The lookup only records Flood Zones 2 and 3, so an empty list means the
    site is in Flood Zone 1.

    Args:
        constraints: Result of ``lookup_constraints``, or None.
    """
    if constraints is None:
        return ["No constraint data available for this site; check manually."]

    flood_zones = cast(list[int], constraints.get("flood_risk_level") or [])
    flood = (
        "Flood Zone " + " and ".join(str(zone) for zone in flood_zones)
        if flood_zones
        else "Flood Zone 1 (outside Flood Zones 2 and 3)"
    )
    conservation_area = constraints.get("conservation_area_name")
    listed_grade = constraints.get("listed_building_grade")
    at_risk = constraints.get("heritage_at_risk_name")
    return [
        f"Flood risk: {flood}",
        "Conservation area: "
        + (
            str(conservation_area).replace("_", " ").title()
            if conservation_area
            else "none"
        ),
        f"Green Belt: {'yes' if constraints.get('is_green_belt') else 'no'}",
        f"Listed building: {f'Grade {listed_grade}' if listed_grade else 'no'}",
        f"Heritage at risk: {at_risk or 'none'}",
    ]


def classify_document(name: str) -> DocumentType:
    """Infer a document's role from its filename.

    Args:
        name: Filename stem, e.g. ``consultation_reply_highways``.
    """
    normalized = name.lower().replace("-", "_")
    tokens = set(normalized.split("_"))
    if normalized.startswith(CONSULTATION_REPLY_PREFIX) or "consultation" in tokens:
        return DocumentType.CONSULTEE_REPLY
    if "representations" in tokens:
        return DocumentType.REPRESENTATION
    if {"drawing", "guide"} <= tokens:
        return DocumentType.DRAWING_SCHEDULE
    if {"application", "form"} <= tokens:
        return DocumentType.APPLICATION_FORM
    if tokens & STATEMENT_TOKENS:
        return DocumentType.SUPPORTING_STATEMENT
    if tokens & DRAWING_TOKENS:
        return DocumentType.DRAWING
    if tokens & TECHNICAL_TOKENS:
        return DocumentType.TECHNICAL_REPORT
    return DocumentType.OTHER


def clean_ocr_text(text: str) -> str:
    """Normalise line endings, OCR artefacts and blank-line runs.

    Args:
        text: Raw extracted text.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = SQUARE_METRES_OCR_RE.sub(r"\1 m²", text)
    lines = [line.rstrip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def read_parsed_document(path: Path) -> tuple[str, list[bytes], int]:
    """Return the raw text, images and page count of one extracted JSON file.

    Args:
        path: JSON file written by ``uv run extract``.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    pages = sorted(raw["pages"], key=lambda page: page["page_number"])
    texts = [text for page in pages for text in page["texts"]]
    images = [base64.b64decode(image) for page in pages for image in page["images"]]
    return "\n".join(texts), images, raw["page_count"]


def _load_document(path: Path) -> CaseDocument:
    """Build a ``CaseDocument`` from one extracted JSON file."""
    raw_text, images, page_count = read_parsed_document(path)
    text, pii_masked = mask_pii(clean_ocr_text(raw_text))
    return CaseDocument(
        name=path.stem,
        doc_type=classify_document(path.stem),
        page_count=page_count,
        text=text,
        images=images,
        injection_findings=screen_text(text),
        pii_masked=dict(pii_masked),
    )


def _mark_superseded_drawings(documents: list[CaseDocument]) -> None:
    """Mark drawings that declare themselves superseded.

    Only drawings are checked: a drawing schedule or statement saying
    "superseded" is describing other documents, not itself.
    """
    current_drawings = [
        doc
        for doc in documents
        if doc.doc_type == DocumentType.DRAWING
        and "superseded" not in doc.name.lower()
        and not DRAWING_SUPERSEDED_RE.search(doc.text)
    ]
    for doc in documents:
        if doc.doc_type != DocumentType.DRAWING or doc in current_drawings:
            continue
        doc.status = DocumentStatus.SUPERSEDED
        doc.status_reason = "drawing is marked as superseded"
        # A "<name>_superseded" drawing is replaced by "<name>" when present.
        replacement = re.sub(r"_superseded$", "", doc.name, flags=re.IGNORECASE)
        if any(current.name == replacement for current in current_drawings):
            doc.superseded_by = replacement


def _group_consultee_replies(documents: list[CaseDocument]) -> list[str]:
    """Tag replies from the same consultee with a shared group; return warnings.

    Reply dates are redacted and filename numbering does not reliably show
    order (a ``_2`` reply can be the earlier one), so all replies stay current
    and the latest position is determined from their content downstream.
    """
    groups: dict[str, list[CaseDocument]] = {}
    for doc in documents:
        name = doc.name.lower().replace("-", "_")
        if doc.doc_type != DocumentType.CONSULTEE_REPLY or not name.startswith(
            CONSULTATION_REPLY_PREFIX
        ):
            continue
        match = REPLY_VERSION_RE.match(
            name.removeprefix(CONSULTATION_REPLY_PREFIX).strip("_")
        )
        if match is None:
            continue
        consultee = match.group("consultee")
        groups.setdefault(CONSULTEE_ALIASES.get(consultee, consultee), []).append(doc)

    warnings = []
    for consultee, replies in groups.items():
        if len(replies) < 2:
            continue
        for doc in replies:
            doc.reply_group = consultee or "unnamed_consultee"
        names = ", ".join(doc.name for doc in replies)
        warnings.append(
            f"{names}: several replies from one consultee; order is not given, "
            "so the latest position is determined from content"
        )
    return warnings


def _quarantine_injections(documents: list[CaseDocument]) -> list[str]:
    """Quarantine documents with suspected prompt injection; return warnings."""
    warnings = []
    for doc in documents:
        if not doc.injection_findings:
            continue
        patterns = sorted({finding.pattern for finding in doc.injection_findings})
        doc.status = DocumentStatus.QUARANTINED
        doc.status_reason = (
            "suspected prompt injection (" + ", ".join(patterns) + "); "
            "requires officer review"
        )
        doc.superseded_by = None
        warnings.append(f"{doc.name}: {doc.status_reason}")
    return warnings


def _find_postcode(documents: list[CaseDocument]) -> tuple[str | None, list[str]]:
    """Return the site postcode and any warnings about it.

    The application form is preferred; otherwise the most frequent postcode
    across the pack is used.
    """
    counts: Counter[str] = Counter()
    form_counts: Counter[str] = Counter()
    for doc in documents:
        found = [re.sub(r"\s", "", match) for match in POSTCODE_RE.findall(doc.text)]
        counts.update(found)
        if doc.doc_type == DocumentType.APPLICATION_FORM:
            form_counts.update(found)

    if not counts:
        return None, ["No site postcode found in any document"]

    warnings = []
    if len(counts) > 1:
        warnings.append(f"Several postcodes found: {sorted(counts)}")
    postcode = (form_counts or counts).most_common(1)[0][0]
    return f"{postcode[:-3]} {postcode[-3:]}", warnings


def _render_document(doc: CaseDocument) -> str:
    """Render one document wrapped in tags that mark it as case evidence."""
    image_note = f' images="{len(doc.images)}"' if doc.images else ""
    group_note = f' reply_group="{doc.reply_group}"' if doc.reply_group else ""
    return (
        f'<document name="{doc.name}" type="{doc.doc_type}"{group_note}{image_note}>\n'
        f"{doc.text}\n</document>"
    )


app = typer.Typer(
    name="case-loader",
    help="Inspect how a case's application pack is loaded.",
    add_completion=False,
    no_args_is_help=False,
)


@app.command()
def main(
    case_ids: list[str] = typer.Argument(
        None, help="Case IDs to inspect, e.g. case-001. Defaults to every case."
    ),
) -> None:
    """Print a summary of each case: documents, statuses, constraints, warnings.

    Args:
        case_ids: Case IDs to inspect. Defaults to every case.
    """
    for case_id in case_ids or list_cases():
        bundle = load_case(case_id)
        typer.echo(f"\n=== {bundle.case_id}  postcode={bundle.postcode}")
        typer.echo(f"constraints: {bundle.constraints}")
        for doc in bundle.documents:
            detail = f" -> {doc.superseded_by}" if doc.superseded_by else ""
            typer.echo(
                f"  {doc.status:<11} {doc.doc_type:<20} {doc.name}{detail}"
                f"  ({len(doc.text)} chars, {len(doc.images)} images)"
            )
        for warning in bundle.warnings:
            typer.echo(f"  ! {warning}")


if __name__ == "__main__":
    app()
