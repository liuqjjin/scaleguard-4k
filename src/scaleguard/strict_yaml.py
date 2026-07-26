"""Strict YAML decoding for configuration and immutable lock boundaries."""

from __future__ import annotations

from typing import Any

import yaml


class StrictYAMLError(ValueError):
    """Raised when YAML is malformed or contains ambiguous mapping keys."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe loader that rejects duplicate keys at every mapping depth."""


def _construct_unique_mapping(
    loader: yaml.SafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate mapping key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def loads(document: str | bytes) -> Any:
    """Decode safe YAML while rejecting duplicate mapping keys."""

    try:
        return yaml.load(document, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise StrictYAMLError(str(error)) from error
