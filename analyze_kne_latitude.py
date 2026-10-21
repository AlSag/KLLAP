#!/usr/bin/env python3
"""Unified command for Galactic-latitude analyses of KNe populations.

The public interface has three explicit subcommands while the numerical and
plotting implementations live in domain-named modules:

``population`` analyses one generated population; ``compare-bins`` compares
paired disjoint latitude bins; ``plateau`` studies a low/high latitude cut.
"""

from __future__ import annotations

import argparse


COMMANDS = {
    "population": "Analyse losses within one generated population",
    "compare-bins": "Compare paired disjoint latitude bins",
    "plateau": "Assess stability across low/high latitude cuts",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=COMMANDS, help="Latitude workflow")
    parser.add_argument(
        "arguments", nargs=argparse.REMAINDER,
        help="Arguments passed to the selected workflow; use COMMAND --help",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "population":
        from kne_latitude_population import main as run
    elif args.command == "compare-bins":
        from kne_latitude_comparison import main as run
    else:
        from kne_latitude_plateau import main as run
    run(args.arguments)


if __name__ == "__main__":
    main()
