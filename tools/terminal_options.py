"""Prompt for a utility's existing argparse options when invoked without arguments."""
from __future__ import annotations

import argparse
import sys


def read_options(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Use normal CLI parsing in batch mode and validate interactive input identically."""
    if len(sys.argv) > 1:
        return parser.parse_args()
    tokens = []
    for action in parser._actions:
        if action.dest == "help":
            continue
        flag = action.option_strings[0] if action.option_strings else None
        if isinstance(action, (argparse._StoreTrueAction, argparse.BooleanOptionalAction)):
            default = bool(action.default)
            while True:
                answer = input(f"{action.help} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
                if answer in {"", "y", "n", "yes", "no"}:
                    break
            if answer and (answer in {"y", "yes"}) != default:
                tokens.append(action.option_strings[-1] if default else flag)
            continue
        choices = ", ".join(map(str, action.choices or ()))
        label = f"{action.help or action.dest}" + (f" ({choices})" if choices else "")
        while True:
            value = input(f"{label} [{action.default if action.default is not None else ''}]: ").strip()
            if value or flag or action.nargs == "?" or action.default is not None:
                break
        if value:
            if flag:
                tokens.append(flag)
            tokens.append(value)
    return parser.parse_args(tokens)
