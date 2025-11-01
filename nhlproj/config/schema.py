from typing import Dict, List, Tuple

# Score categories
INFO_ONLY = "InfoOnly"
POINTS = "Points"
BANGER = "Banger"
OTHER = "Other"

# NST schema as provided by user (column header -> ScoreCat)
NST_SCHEMA: Dict[str, str] = {
    "Unnamed": INFO_ONLY,
    "Player": INFO_ONLY,
    "Team": INFO_ONLY,
    "Position": INFO_ONLY,
    "GP": INFO_ONLY,
    "TOI": INFO_ONLY,
    "TOI/GP": INFO_ONLY,
    "Goals/60": POINTS,
    "Total Assists/60": POINTS,
    "First Assists/60": OTHER,
    "Second Assists/60": OTHER,
    "Total Points/60": OTHER,
    "IPP": OTHER,
    "Shots/60": POINTS,
    "SH%": OTHER,
    "ixG/60": OTHER,
    "iCF/60": OTHER,
    "iFF/60": OTHER,
    "iSCF/60": OTHER,
    "iHDCF/60": OTHER,
    "Rush Attempts/60": OTHER,
    "Rebounds Created/60": OTHER,
    "PIM/60": BANGER,
    "Total Penalties/60": OTHER,
    "Minor/60": OTHER,
    "Major/60": OTHER,
    "Misconduct/60": OTHER,
    "Penalties Drawn/60": OTHER,
    "Giveaways/60": OTHER,
    "Takeaways/60": OTHER,
    "Hits/60": BANGER,
    "Hits Taken/60": OTHER,
    "Shots Blocked/60": BANGER,
    "Faceoffs Won/60": POINTS,
    "Faceoffs Lost/60": OTHER,
    "Faceoffs %": OTHER,
}

# Synonyms mapping for columns that may appear under different spellings on NST
# Keys are possible incoming names; values are the canonical names used above.
NST_SYNONYMS: Dict[str, str] = {
    # shots
    "iSF/60": "Shots/60",
    "iSh/60": "Shots/60",
    "Shot Attempts/60": "iCF/60",
    # penalties/hits/blocks
    "Hits Taken/60": "Hits Taken/60",
    "Blocks/60": "Shots Blocked/60",
    # faceoffs
    "FO Won/60": "Faceoffs Won/60",
    "FO Lost/60": "Faceoffs Lost/60",
    "FO%": "Faceoffs %",
}


def split_columns_by_scorecat(columns: List[str], schema: Dict[str, str]) -> Tuple[List[str], List[str], List[str]]:
    """Given a list of DataFrame columns and a schema, return three lists:
    - points columns present in df
    - banger columns present in df
    - columns to analyze (points + banger), preserving order from incoming list
    InfoOnly/Other are excluded from the analyze set.
    """
    points: List[str] = []
    banger: List[str] = []
    for c in columns:
        cat = schema.get(c)
        if cat == POINTS:
            points.append(c)
        elif cat == BANGER:
            banger.append(c)
    analyze = points + banger
    return points, banger, analyze
