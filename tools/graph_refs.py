import re
from typing import Any


CHARACTER_REF_PREFIX = "c_"
ENVIRONMENT_REF_PREFIX = "e_"
PLOT_REF_PREFIX = "p_"

CHARACTER_TEMP_PREFIX = "c_tmp_"
ENVIRONMENT_TEMP_PREFIX = "e_tmp_"
PLOT_TEMP_PREFIX = "p_tmp_"


_CHARACTER_REAL_RE = re.compile(r"^c_([1-9]\d*)$")
_ENVIRONMENT_REAL_RE = re.compile(r"^e_([1-9]\d*)$")
_PLOT_REAL_RE = re.compile(r"^p_([1-9]\d*)$")
_CHARACTER_TEMP_RE = re.compile(r"^c_tmp_\d+$")
_ENVIRONMENT_TEMP_RE = re.compile(r"^e_tmp_\d+$")
_PLOT_TEMP_RE = re.compile(r"^p_tmp_\d+$")


def _is_non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def make_character_ref(value: int | str) -> str:
    if isinstance(value, bool):
        raise ValueError("character id must be a positive integer, not bool")
    if isinstance(value, str):
        if _CHARACTER_REAL_RE.fullmatch(value):
            return value
        raise ValueError(f"character ref must be canonical c_K: {value!r}")
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"character id must be a positive integer: {value!r}")
    return f"{CHARACTER_REF_PREFIX}{value}"


def make_environment_ref(value: int | str) -> str:
    if isinstance(value, bool):
        raise ValueError("environment id must be a positive integer, not bool")
    if isinstance(value, str):
        if _ENVIRONMENT_REAL_RE.fullmatch(value):
            return value
        raise ValueError(f"environment ref must be canonical e_K: {value!r}")
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"environment id must be a positive integer: {value!r}")
    return f"{ENVIRONMENT_REF_PREFIX}{value}"


def make_plot_ref(value: int | str) -> str:
    if isinstance(value, bool):
        raise ValueError("plot id must be a positive integer, not bool")
    if isinstance(value, str):
        if _PLOT_REAL_RE.fullmatch(value):
            return value
        raise ValueError(f"plot ref must be canonical p_K: {value!r}")
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"plot id must be a positive integer: {value!r}")
    return f"{PLOT_REF_PREFIX}{value}"


def make_temp_ref(prefix: str, index: int) -> str:
    return f"{prefix}{index:02d}"


def is_character_ref(value: Any) -> bool:
    return isinstance(value, str) and bool(_CHARACTER_REAL_RE.fullmatch(value) or _CHARACTER_TEMP_RE.fullmatch(value))


def is_environment_ref(value: Any) -> bool:
    return isinstance(value, str) and bool(_ENVIRONMENT_REAL_RE.fullmatch(value) or _ENVIRONMENT_TEMP_RE.fullmatch(value))


def is_plot_ref(value: Any) -> bool:
    return isinstance(value, str) and bool(_PLOT_REAL_RE.fullmatch(value) or _PLOT_TEMP_RE.fullmatch(value))


def is_real_character_ref(value: Any) -> bool:
    return isinstance(value, str) and bool(_CHARACTER_REAL_RE.fullmatch(value))


def is_real_environment_ref(value: Any) -> bool:
    return isinstance(value, str) and bool(_ENVIRONMENT_REAL_RE.fullmatch(value))


def is_real_plot_ref(value: Any) -> bool:
    return isinstance(value, str) and bool(_PLOT_REAL_RE.fullmatch(value))


def is_temp_character_ref(value: Any) -> bool:
    return isinstance(value, str) and bool(_CHARACTER_TEMP_RE.fullmatch(value))


def is_temp_environment_ref(value: Any) -> bool:
    return isinstance(value, str) and bool(_ENVIRONMENT_TEMP_RE.fullmatch(value))


def is_temp_plot_ref(value: Any) -> bool:
    return isinstance(value, str) and bool(_PLOT_TEMP_RE.fullmatch(value))


def parse_character_ref(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if not isinstance(value, str):
        return None
    match = _CHARACTER_REAL_RE.fullmatch(value)
    if not match:
        return None
    return int(match.group(1))


def parse_environment_ref(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if not isinstance(value, str):
        return None
    match = _ENVIRONMENT_REAL_RE.fullmatch(value)
    if not match:
        return None
    return int(match.group(1))


def parse_plot_ref(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if not isinstance(value, str):
        return None
    match = _PLOT_REAL_RE.fullmatch(value)
    if not match:
        return None
    return int(match.group(1))


def normalize_character_ref(value: Any, allow_temp: bool = True) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        if value <= 0:
            return None
        return make_character_ref(value)
    if not isinstance(value, str):
        return None
    if _CHARACTER_REAL_RE.fullmatch(value):
        return value
    if allow_temp and _CHARACTER_TEMP_RE.fullmatch(value):
        return value
    return None


def normalize_environment_ref(value: Any, allow_temp: bool = True) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        if value <= 0:
            return None
        return make_environment_ref(value)
    if not isinstance(value, str):
        return None
    if _ENVIRONMENT_REAL_RE.fullmatch(value):
        return value
    if allow_temp and _ENVIRONMENT_TEMP_RE.fullmatch(value):
        return value
    return None


def normalize_plot_ref(value: Any, allow_temp: bool = True) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        if value <= 0:
            return None
        return make_plot_ref(value)
    if not isinstance(value, str):
        return None
    if _PLOT_REAL_RE.fullmatch(value):
        return value
    if allow_temp and _PLOT_TEMP_RE.fullmatch(value):
        return value
    return None


def normalize_temp_id(value: Any, prefix: str, index: int) -> str:
    if _is_non_empty_text(value):
        return value.strip()
    return make_temp_ref(prefix, index)
