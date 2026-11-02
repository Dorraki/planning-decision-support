"""Material considerations and the policies assessed against each one.

Each case is assessed consideration by consideration. This module lists the
considerations and maps each one to the Local Plan policies and NPPF
paragraphs given to the assessor, keyed by the citation IDs in
``policy_index``.
"""

from enum import StrEnum


class Consideration(StrEnum):
    """Material considerations that recur in Doncaster householder decisions."""

    PRINCIPLE = "principle_of_development"
    DESIGN_AND_CHARACTER = "design_and_character"
    HERITAGE = "heritage"
    NEIGHBOUR_AMENITY = "neighbour_amenity"
    LIVING_STANDARDS = "living_standards"
    HIGHWAYS_AND_PARKING = "highways_and_parking"
    FLOOD_RISK = "flood_risk"
    DRAINAGE = "drainage"
    ECOLOGY_AND_TREES = "ecology_and_trees"
    POLLUTION = "pollution_and_contamination"


# Starting candidates per consideration: the policies cited in the
# development-set decisions, plus core NPPF and Local Plan policies for topics
# those cases did not cover (e.g. heritage, ecology). The assessor still
# decides which ones apply.
CONSIDERATION_POLICY_IDS: dict[Consideration, tuple[str, ...]] = {
    Consideration.PRINCIPLE: (
        "LP1",
        "LP9",
        "LP10",
        "LP22",
        "LP23",
        "LP24",
        "LP25",
        "LP50",
        "LP51",
        "NPPF11",
        "NPPF88",
        "NPPF89",
        "NPPF124",
    ),
    Consideration.DESIGN_AND_CHARACTER: (
        "LP41",
        "LP42",
        "LP44",
        "LP46",
        "NPPF131",
        "NPPF135",
        "NPPF139",
    ),
    Consideration.HERITAGE: (
        "LP34",
        "LP36",
        "LP37",
        "LP40",
        "NPPF207",
        "NPPF212",
        "NPPF213",
        "NPPF215",
    ),
    Consideration.NEIGHBOUR_AMENITY: ("LP10", "LP44", "LP54", "NPPF135", "NPPF198"),
    Consideration.LIVING_STANDARDS: ("LP9", "LP10", "LP44", "LP45", "NPPF135"),
    Consideration.HIGHWAYS_AND_PARKING: (
        "LP13",
        "LP16",
        "LP42",
        "NPPF115",
        "NPPF116",
        "NPPF117",
    ),
    Consideration.FLOOD_RISK: (
        "LP57",
        "NPPF170",
        "NPPF173",
        "NPPF175",
        "NPPF176",
        "NPPF181",
    ),
    Consideration.DRAINAGE: ("LP56", "NPPF181", "NPPF182"),
    Consideration.ECOLOGY_AND_TREES: ("LP29", "LP30", "LP32", "NPPF187", "NPPF193"),
    Consideration.POLLUTION: ("LP54", "LP55", "NPPF196", "NPPF198"),
}
