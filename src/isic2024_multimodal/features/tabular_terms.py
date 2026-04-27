from __future__ import annotations


STRICT_RAW_NUMERIC = "strict_raw_numeric"
STRICT_BASE = "strict_base"
STRICT_FE = "strict_fe"
STRICT_MAIN_INPUT = "strict_main_input"
RELAXED = "relaxed"
ORACLE = "oracle"

LEGACY_STRICT = "strict"
LEGACY_STRICT_FULL = "strict_full"

FEATURE_SET_ALIASES = {
    LEGACY_STRICT: STRICT_MAIN_INPUT,
    LEGACY_STRICT_FULL: STRICT_MAIN_INPUT,
    STRICT_BASE: STRICT_BASE,
    STRICT_FE: STRICT_FE,
    STRICT_MAIN_INPUT: STRICT_MAIN_INPUT,
    RELAXED: RELAXED,
    ORACLE: ORACLE,
}

FEATURE_SET_DISPLAY_NAMES = {
    STRICT_BASE: "Strict Base",
    STRICT_FE: "Strict FE",
    STRICT_MAIN_INPUT: "Strict Main Input",
    RELAXED: "Relaxed",
    ORACLE: "Oracle",
}

FEATURE_SET_DISPLAY_ORDER = {
    STRICT_BASE: 0,
    STRICT_FE: 1,
    STRICT_MAIN_INPUT: 2,
    RELAXED: 3,
    ORACLE: 4,
}


def normalize_feature_set_name(name: str) -> str:
    return FEATURE_SET_ALIASES.get(name, name)


def normalize_feature_set_names(names: list[str] | None) -> list[str] | None:
    if names is None:
        return None
    return [normalize_feature_set_name(name) for name in names]


def feature_set_display_name(name: str) -> str:
    normalized = normalize_feature_set_name(name)
    return FEATURE_SET_DISPLAY_NAMES.get(normalized, normalized.replace("_", " ").title())
