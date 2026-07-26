"""Strict JSON decoding for evidence and protocol boundaries."""

from __future__ import annotations

import json
from typing import Any, NoReturn, cast


class StrictJSONError(ValueError):
    """Raised when JSON is ambiguous, non-standard, or malformed."""


def _reject_constant(value: str) -> NoReturn:
    raise StrictJSONError(f"non-standard JSON constant {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def loads(document: str | bytes | bytearray) -> Any:
    """Decode standard JSON and reject duplicate keys at every object depth."""

    try:
        return json.loads(
            document,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except StrictJSONError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise StrictJSONError(str(error)) from error


def loads_object(document: str | bytes | bytearray) -> dict[str, Any]:
    """Decode a JSON document whose root must be an object."""

    value = loads(document)
    if not isinstance(value, dict):
        raise StrictJSONError("JSON document must be an object")
    return cast(dict[str, Any], value)
