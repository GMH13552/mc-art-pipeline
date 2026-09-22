"""Tests for the CLI's own argument plumbing.

The family path never reads --size, so a double parse there stayed hidden until
a plain single-asset run hit it.
"""

from __future__ import annotations

from studio_next.cli import _parse_size, build_parser


def test_parse_size_is_idempotent() -> None:
    """argparse already parsed the default, so the handler parses a tuple."""
    assert _parse_size("16x16") == (16, 16)
    assert _parse_size((16, 16)) == (16, 16)
    assert _parse_size((64, 32)) == (64, 32)
    assert _parse_size(None) == (16, 16)


def test_the_parser_default_is_already_a_tuple() -> None:
    args = build_parser().parse_args(["generate", "--query", "x", "--source", "/tmp"])
    assert args.size == (16, 16)
    assert _parse_size(args.size) == (16, 16)


def test_an_explicit_size_survives_the_round_trip() -> None:
    args = build_parser().parse_args(["generate", "--query", "x", "--source", "/tmp", "--size", "64x32"])
    assert args.size == (64, 32)
    assert _parse_size(args.size) == (64, 32)
