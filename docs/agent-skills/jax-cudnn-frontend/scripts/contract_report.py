#!/usr/bin/env python3
"""Kernel contract report: signatures, in-body constraints, and vendor call
sites for a cuDNN Frontend / CuTe DSL kernel class or function.

This automates the "Rosetta stone" step: the vendor's own call sites (and
especially its buffer ALLOCATIONS near those call sites) are the ground truth
for output shapes, argument order, and workspace layout.

Usage:
    python contract_report.py cudnn.block_sparse_attention.csrc.fwd.sm100_blk64.bsa_fwd_sm100.BlockSparseAttnForwardSm100Blk64
    python contract_report.py <dotted.path.Symbol> [--package cudnn] [--context 8]

Notes:
 - Pure introspection; never executes the kernel.
 - Output is evidence, not interpretation: read the printed assert/remap
   lines yourself — comments in signatures can contradict the code.
"""
import argparse
import importlib
import inspect
import os
import re
import sys

CONSTRAINT_PAT = re.compile(
    r"^\s*(assert\b|.*\bcheck_dim\(|.*element_type\b|.*const_expr\(|"
    r".*raise\b|.*\.launch\()"
)
ALLOC_PAT = re.compile(r"(empty|zeros|ones|full|empty_like|zeros_like)\s*\(")


def resolve(dotted):
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        modname = ".".join(parts[:i])
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        obj = mod
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        return obj, mod
    raise SystemExit(f"could not resolve {dotted!r} — check spelling and that "
                     f"the package is installed (see inspect_environment.py)")


def show_signature(obj, name):
    print(f"\n-- {name} signature " + "-" * 40)
    try:
        print(f"{name}{inspect.signature(obj)}")
    except (TypeError, ValueError):
        print("<signature unavailable via inspect>")
    # Raw source of the def line(s) preserves parameter comments, which often
    # carry (possibly wrong!) shape hints — print them labeled as hints.
    try:
        src = inspect.getsource(obj)
        header = []
        depth = 0
        for line in src.splitlines():
            header.append(line)
            depth += line.count("(") - line.count(")")
            if depth <= 0 and header:
                break
        print("raw def (comments are HINTS, code below is CONTRACT):")
        for line in header[:40]:
            print(f"  {line}")
    except (OSError, TypeError):
        pass


def show_constraints(obj):
    print("\n-- in-body constraints (asserts / check_dim / dtype / launches) --")
    try:
        src, start = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        print("<source unavailable>")
        return
    hits = 0
    for off, line in enumerate(src):
        if CONSTRAINT_PAT.match(line):
            print(f"  L{start + off}: {line.rstrip()}")
            hits += 1
            if hits >= 60:
                print("  ... (truncated)")
                break
    if not hits:
        print("  (none matched — read __call__ manually; contracts may be "
              "enforced in helpers)")
    n_launch = sum(1 for l in src if ".launch(" in l)
    print(f"  internal .launch( count: {n_launch}"
          + ("  <- multi-launch body: exercise integration bridges carefully"
                 if n_launch > 1 else ""))


def show_call_sites(symbol_name, package, context):
    print(f"\n-- vendor call sites for '{symbol_name}' in package "
          f"'{package}' --")
    try:
        pkg = importlib.import_module(package)
        root = os.path.dirname(pkg.__file__)
    except Exception as e:  # noqa: BLE001
        print(f"<cannot locate package {package}: {e}>")
        return
    found = 0
    for dirpath, _dirs, files in os.walk(root):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(dirpath, fname)
            try:
                with open(path, errors="replace") as f:
                    lines = f.readlines()
            except OSError:
                continue
            for i, line in enumerate(lines):
                if symbol_name in line and f"class {symbol_name}" not in line \
                        and f"def {symbol_name}" not in line:
                    found += 1
                    rel = os.path.relpath(path, root)
                    print(f"\n  {rel}:{i + 1}")
                    lo, hi = max(0, i - context), min(len(lines), i + context + 1)
                    for j in range(lo, hi):
                        mark = ">>" if j == i else "  "
                        print(f"   {mark} {j + 1}: {lines[j].rstrip()}")
                    # nearby allocations = candidate buffer contracts
                    allocs = [f"      L{j + 1}: {lines[j].strip()}"
                              for j in range(max(0, i - 60),
                                             min(len(lines), i + 60))
                              if ALLOC_PAT.search(lines[j])]
                    if allocs:
                        print("    nearby allocations (BUFFER-SHAPE GROUND TRUTH"
                              " candidates):")
                        for a in allocs[:12]:
                            print(a)
                    if found >= 8:
                        print("\n  ... (more call sites exist; refine manually)")
                        return
    if not found:
        print("  none found — the kernel may be exercised only from tests or "
              "another package; widen the search (grep site-packages).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dotted", help="dotted path to kernel class/function")
    ap.add_argument("--package", default=None,
                    help="package to scan for call sites (default: top-level "
                         "of the dotted path)")
    ap.add_argument("--context", type=int, default=8)
    args = ap.parse_args()

    obj, mod = resolve(args.dotted)
    name = args.dotted.rsplit(".", 1)[-1]
    print(f"resolved : {obj!r}")
    print(f"defined  : {getattr(mod, '__file__', '?')}")

    if inspect.isclass(obj):
        show_signature(obj.__init__, f"{name}.__init__")
        call = getattr(obj, "__call__", None)
        if call is not None and call is not object.__call__:
            show_signature(call, f"{name}.__call__")
            show_constraints(call)
    else:
        show_signature(obj, name)
        show_constraints(obj)

    package = args.package or args.dotted.split(".")[0]
    show_call_sites(name, package, args.context)

    print("\n-- next steps (see references/contract-discovery.md) --")
    print(" 1. Fill the contract checklist; every buffer shape must come from")
    print("    a vendor allocation, not a comment or a name.")
    print(" 2. Where a signature comment and an assert/remap disagree, the")
    print("    code wins; record the discrepancy.")
    print(" 3. Repeat for EACH architecture variant (sm90/sm100/...): ctor")
    print("    params, arg order, and optionality all may differ.")


if __name__ == "__main__":
    main()
