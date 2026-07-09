import re


def normalize_taxon_name(name: str) -> str:
    cleaned = name.strip()
    cleaned = re.sub(r"^[kgpcofst]__", "", cleaned)
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\s*;\s*$", "", cleaned)
    return cleaned


def clean_taxon_string(name: str) -> str:
    return normalize_taxon_name(name).strip().lower()


def extract_genus(name: str) -> str | None:
    parts = name.split(";")
    for part in parts:
        part = part.strip()
        if part.startswith("g__"):
            return normalize_taxon_name(part)
    normalized = normalize_taxon_name(name)
    return normalized.split()[0] if " " in normalized else normalized


def extract_species(name: str) -> str | None:
    parts = name.split(";")
    for part in parts:
        part = part.strip()
        if part.startswith("s__"):
            return normalize_taxon_name(part)
    normalized = normalize_taxon_name(name)
    words = normalized.split()
    if len(words) >= 2:
        return normalized
    return None
