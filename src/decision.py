"""Decision schema: the draft decision the pipeline produces for a case."""

from enum import StrEnum

from pydantic import BaseModel, Field


class DecisionType(StrEnum):
    """The two outcomes an officer can issue."""

    APPROVE = "approve"
    REFUSE = "refuse"


class Decision(BaseModel):
    """A draft decision with its reasons and, for an approval, conditions."""

    decision: DecisionType = Field(..., description="Approve or refuse")
    reasons: list[str] = Field(
        default_factory=list, description="Planning reasons for the decision"
    )
    conditions: list[str] = Field(
        default_factory=list, description="Conditions attached to an approval"
    )

    def to_markdown(self, case_id: str, local_authority: str | None = None) -> str:
        """Return the decision as a markdown notice.

        Args:
            case_id: Case the decision is for.
            local_authority: Authority issuing the decision, if known.
        """
        lines = [f"# Planning Decision: {self.decision.value.title()}", ""]
        lines.append(f"- **Case:** {case_id}")
        if local_authority:
            lines.append(f"- **Local Authority:** {local_authority}")
        lines.append("")
        if self.reasons:
            heading = (
                "Reasons for Refusal"
                if self.decision == DecisionType.REFUSE
                else "Reasons"
            )
            lines += [f"## {heading}", "", *(f"- {r}" for r in self.reasons), ""]
        if self.conditions:
            lines += ["## Conditions", "", *(f"- {c}" for c in self.conditions), ""]
        return "\n".join(lines).rstrip()
