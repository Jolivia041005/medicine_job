import re

NAMED_GROUP_PATTERNS = [
    re.compile(r"UCG-\d+", re.IGNORECASE),
    re.compile(r"R-\d+\s*group", re.IGNORECASE),
    re.compile(r"\w+\s+group", re.IGNORECASE),
]


def normalize_taxon(name: str) -> str:
    cleaned = str(name).strip()
    cleaned = re.sub(r"^[kgpcofst]__", "", cleaned)
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\s*;\s*$", "", cleaned)
    return cleaned


def clean_taxon(name: str) -> str:
    return normalize_taxon(name).strip().lower()


def parse_taxonomy_string(taxon_str: str) -> dict[str, str | None]:
    taxon_str = str(taxon_str)
    parts = [p.strip() for p in taxon_str.split(";") if p.strip()]
    result: dict[str, str | None] = {
        "full": taxon_str,
        "species": None,
        "genus": None,
        "family": None,
        "order": None,
        "class": None,
        "phylum": None,
    }

    for part in parts:
        if part.startswith("s__") and part[3:]:
            result["species"] = normalize_taxon(part)
        elif part.startswith("g__") and part[3:]:
            result["genus"] = normalize_taxon(part)
        elif part.startswith("f__") and part[3:]:
            result["family"] = normalize_taxon(part)
        elif part.startswith("o__") and part[3:]:
            result["order"] = normalize_taxon(part)
        elif part.startswith("c__") and part[3:]:
            result["class"] = normalize_taxon(part)
        elif part.startswith("p__") and part[3:]:
            result["phylum"] = normalize_taxon(part)

    if result["species"] is None and result["genus"] is None:
        words = normalize_taxon(taxon_str).split()
        if len(words) >= 2:
            maybe_species = f"{words[0]} {words[1]}"
            if not any(p in maybe_species.lower() for p in ["bacteria", "archaea"]):
                result["species"] = maybe_species
                result["genus"] = words[0]
        elif len(words) == 1:
            result["genus"] = words[0]

    return result


def extract_genus_name(taxon_str: str) -> str | None:
    parsed = parse_taxonomy_string(taxon_str)
    return parsed["genus"]


def extract_species_name(taxon_str: str) -> str | None:
    parsed = parse_taxonomy_string(taxon_str)
    return parsed["species"]


def extract_family_name(taxon_str: str) -> str | None:
    parsed = parse_taxonomy_string(taxon_str)
    if parsed["family"]:
        return parsed["family"]
    genus = parsed["genus"]
    if genus and (genus.endswith("aceae") or genus.endswith("aceae group")):
        return genus
    return None


def is_named_group(name: str) -> bool:
    return any(p.search(name) for p in NAMED_GROUP_PATTERNS)
