"""Site constraints (flood zone, conservation area, Green Belt, listing) by postcode.

A real deployment would query the council's GIS layers with the site's
coordinates. Here constraints come from an optional JSON file keyed by
postcode, so the pipeline can run without a GIS service:

    {
      "DN1 1AA": {
        "flood_risk_level": [2, 3],
        "conservation_area_name": "doncaster_high_street",
        "is_green_belt": false,
        "listed_building_grade": null,
        "heritage_at_risk_name": null
      }
    }

An unknown postcode returns None, and the pipeline tells the officer to check
constraints manually rather than assuming there are none.
"""

import json
from pathlib import Path

CONSTRAINTS_PATH = Path("data") / "site_constraints.json"


def lookup_constraints(
    postcode: str, path: Path = CONSTRAINTS_PATH
) -> dict[str, object] | None:
    """Return the recorded constraints for a postcode, or None if unknown.

    Args:
        postcode: Site postcode; spacing and case are ignored.
        path: JSON file of constraints keyed by postcode.
    """
    if not path.exists():
        return None
    records: dict[str, dict[str, object]] = json.loads(path.read_text("utf-8"))
    by_key = {_key(code): record for code, record in records.items()}
    return by_key.get(_key(postcode))


def _key(postcode: str) -> str:
    """Normalise a postcode for lookup, e.g. 'dn1 1aa' -> 'DN11AA'."""
    return postcode.replace(" ", "").upper()
