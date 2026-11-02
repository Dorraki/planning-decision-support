"""Rule-based check that a pack contains the evidence policy requires.

Mirrors a council's validation checklist, without a model: required documents
are checked against the site's designations. Missing evidence does not stop the
assessment (the draft must still approve or refuse), but it raises an
officer alert listing what to request before the draft is relied on.

Example: a driveway gate in Flood Zones 2 and 3 arrived with no flood risk
assessment. The pipeline approved it, but this check flags that NPPF paragraph
181 requires one, so the approval should not be relied on until it is provided.
"""

from typing import cast

from case_loader import CaseBundle, DocumentStatus, DocumentType, OfficerAlert

INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def missing_evidence(bundle: CaseBundle) -> list[str]:
    """Return the evidence a pack lacks, each with the requirement behind it.

    Args:
        bundle: Loaded case.
    """
    current = bundle.documents_with_status(DocumentStatus.CURRENT)
    types = {doc.doc_type for doc in current}
    missing = []

    if DocumentType.APPLICATION_FORM not in types:
        missing.append("Application form: none among the current documents.")
    if DocumentType.DRAWING not in types:
        missing.append("Plans: no current drawings of the proposal.")
    missing += [
        f"{doc.name}: no readable text or images; obtain a legible copy."
        for doc in current
        if not doc.text.strip() and not doc.images
    ]

    if bundle.constraints is None:
        missing.append(
            "Site constraints: the site postcode was not found in the geospatial "
            "lookup, so flood zone, conservation area and listed status are unverified."
        )
        return missing

    flood_zones = cast(list[int], bundle.constraints.get("flood_risk_level") or [])
    has_flood_assessment = any(
        doc.doc_type == DocumentType.TECHNICAL_REPORT and "flood" in doc.name.lower()
        for doc in current
    )
    if flood_zones and not has_flood_assessment:
        zones = " and ".join(str(zone) for zone in flood_zones)  # [2, 3] to "2 and 3"
        missing.append(
            f"Flood risk assessment: the site is in Flood Zone {zones}, and NPPF "
            "paragraph 181 (footnote 63) requires a site-specific flood risk "
            "assessment for all development in Flood Zones 2 and 3 (Local Plan "
            "Policy 57)."
        )

    conservation_area = bundle.constraints.get("conservation_area_name")
    listed_grade = bundle.constraints.get("listed_building_grade")
    has_heritage_statement = any("heritage" in doc.name.lower() for doc in current)
    if (conservation_area or listed_grade) and not has_heritage_statement:
        asset = (
            f"a Grade {listed_grade} listed building"
            if listed_grade
            else f"the {str(conservation_area).replace('_', ' ').title()} "
            "Conservation Area"
        )
        missing.append(
            f"Heritage statement: the site affects {asset}, and NPPF paragraph 207 "
            "requires the applicant to describe the significance of affected "
            "heritage assets. Check whether another document covers this."
        )
    return missing


def evidence_alert(bundle: CaseBundle) -> OfficerAlert | None:
    """Return an officer alert if required evidence is missing, else None.

    Args:
        bundle: Loaded case.
    """
    missing = missing_evidence(bundle)
    if not missing:
        return None
    return OfficerAlert(
        issue=INSUFFICIENT_EVIDENCE,
        detail=(
            "The pack may not contain evidence that policy requires before "
            "determination. This draft was still produced, but request or check "
            "the following before relying on it:"
        ),
        items=missing,
    )
