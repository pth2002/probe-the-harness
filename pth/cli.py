"""Command line entry point: ``pth <command> ...``. Exits with status 1 when any check fails."""
from __future__ import annotations

import argparse
import csv
import fnmatch
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

from .batches import check_batch_ids
from .config import check_arms
from .paired import paired_differences, read_results_csv
from .report import Report
from . import verl


def _levels(args) -> Sequence[str]:
    return ("fail", "warn") if args.quiet else ("fail", "warn", "info")


def _use_color() -> bool:
    if not sys.stdout.isatty() or "NO_COLOR" in os.environ or os.environ.get("TERM") == "dumb":
        return False
    if os.name == "nt":  # enable ANSI escapes on the Windows console
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
        except Exception:
            return False
    return True


def _show(rep: Report, args) -> None:
    print(rep.format(_levels(args), color=_use_color()))


def _cmd_verl(args) -> int:
    arms: Dict[str, str] = {}
    paths: List[str] = list(args.logs)
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm expects NAME=GLOB, got {spec!r}")
        name, pattern = spec.split("=", 1)
        matched = sorted(glob.glob(pattern)) or [
            p for p in args.logs if fnmatch.fnmatch(p, pattern) or fnmatch.fnmatch(os.path.basename(p), pattern)
        ]
        if not matched:
            raise SystemExit(f"--arm {name}: no log matches {pattern!r}")
        for p in matched:
            if p in arms and arms[p] != name:
                raise SystemExit(f"{p} matches both arm {arms[p]!r} and arm {name!r}")
            arms[p] = name
            if p not in paths:
                paths.append(p)
    if not paths:
        raise SystemExit("no logs given")

    failed = False
    for p in paths:
        rep = verl.check_log(p, kl_tol=args.kl_tol)
        rep.subject = f"{p}" + (f"  [arm {arms[p]}]" if p in arms else "")
        _show(rep, args)
        print()
        failed |= not rep.ok

    if arms:
        names = {p: os.path.basename(p) for p in arms}
        if len(set(names.values())) < len(names):
            names = {p: p for p in arms}
        configs, arm_of = {}, {}
        for p in arms:
            cfg = verl.read_config(p)
            if cfg is None:
                print(f"check 2: no configuration in {p}, left out of the arm comparison\n")
            else:
                configs[names[p]] = cfg
                arm_of[names[p]] = arms[p]
        rep = check_arms(configs, arm_of, vary=args.vary, replicate=args.replicate)
        _show(rep, args)
        print()
        failed |= not rep.ok
    return 1 if failed else 0


def _read_ids(path: str, column: Optional[str]) -> List[str]:
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return [str(json.loads(line)[column or "batch"]) for line in f if line.strip()]
    if path.endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            return [row[column or "batch"] for row in csv.DictReader(f)]
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _cmd_batches(args) -> int:
    rep = check_batch_ids(_read_ids(args.file, args.column), expect_unique=not args.allow_repeats)
    _show(rep, args)
    return 0 if rep.ok else 1


def _cmd_paired(args) -> int:
    results = read_results_csv(args.file)
    rep = paired_differences(results, args.a, args.b, scale=args.scale)
    _show(rep, args)
    return 0 if rep.ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pth", description="Probe the Harness: setup checks for RL comparisons.")
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("verl", help="checks 1 and 2 on verl console logs")
    v.add_argument("logs", nargs="*", help="verl console logs")
    v.add_argument("--arm", action="append", default=[], metavar="NAME=GLOB",
                   help="assign logs to an arm, once per arm, to add check 2 across arms")
    v.add_argument("--vary", action="append", default=[], metavar="KEY",
                   help="config field the comparison varies between arms (repeatable)")
    v.add_argument("--replicate", action="append", default=None, metavar="KEY",
                   help="config field that distinguishes replicates (default: data.seed)")
    v.add_argument("--kl-tol", type=float, default=0.05,
                   help="sampler-learner KL above which the sampler counts as lagging (default 0.05)")
    v.add_argument("-q", "--quiet", action="store_true", help="show only failures and warnings")
    v.set_defaults(func=_cmd_verl)

    b = sub.add_parser("batches", help="check 3 on a list of batch IDs, one per update")
    b.add_argument("file", help=".txt (one ID per line), .csv or .jsonl")
    b.add_argument("--column", help="column or key holding the batch ID (default: batch)")
    b.add_argument("--allow-repeats", action="store_true", help="the design reuses batches on purpose")
    b.add_argument("-q", "--quiet", action="store_true")
    b.set_defaults(func=_cmd_batches)

    c = sub.add_parser("paired", help="check 6 on a CSV with columns method,seed,value")
    c.add_argument("file")
    c.add_argument("a", help="first method")
    c.add_argument("b", help="second method")
    c.add_argument("--scale", type=float, default=1.0, help="multiply differences, e.g. 100 for points")
    c.add_argument("-q", "--quiet", action="store_true")
    c.set_defaults(func=_cmd_paired)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "replicate", "unset") is None:
        args.replicate = ["data.seed"]
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
