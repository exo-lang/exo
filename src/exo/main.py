from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import sys

from pathlib import Path

import exo

from contextlib import contextmanager


@contextmanager
def pythonpath(path: Path):
    try:
        sys.path.insert(0, str(path))
        yield
    finally:
        sys.path = sys.path[1:]


def exocc(*args, name="exocc"):
    sys.setrecursionlimit(10000)

    parser = argparse.ArgumentParser(prog=name, description="Compile an Exo library.")
    parser.add_argument(
        "-o",
        "--outdir",
        metavar="OUTDIR",
        help="output directory for build artifacts",
    )
    parser.add_argument(
        "-p",
        "--pythonpath",
        metavar="PYTHONPATH",
        help=(
            "directory to add to PYTHONPATH. Defaults to parent of a single source "
            "file or the current working directory for multiple source files."
        ),
        type=Path,
    )
    parser.add_argument("-s", "--stem", help="base name for .c and .h files")
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s version {exo.__version__}",
        help="print the version and exit",
    )
    parser.add_argument(
        "source", type=Path, nargs="+", help="source file(s) to compile"
    )

    args = parser.parse_args(args)
    srcname = args.source[0].stem

    if not args.outdir:
        if len(args.source) == 1:
            outdir = Path(srcname)
        else:
            parser.error("Must provide -o when processing multiple source files.")
    else:
        outdir = Path(args.outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    if not args.stem:
        if len(args.source) == 1:
            args.stem = srcname
        else:
            parser.error("Must provide --stem when processing multiple source files.")

    if not args.pythonpath:
        if len(args.source) == 1:
            args.pythonpath = args.source[0].parent
        else:
            args.pythonpath = Path(".")

    with pythonpath(args.pythonpath):
        library = [
            proc
            for mod in args.source
            for proc in get_procs_from_module(load_user_code(mod))
        ]

    exo.compile_procs(library, outdir, f"{args.stem}.c", f"{args.stem}.h")
    write_depfile(outdir, args.stem)


def write_depfile(outdir, stem):
    modules = set()
    for mod in sys.modules.values():
        try:
            modules.add(inspect.getfile(mod))
        except TypeError:
            pass  # this is the case for built-in modules

    c_file = outdir / f"{stem}.c"
    h_file = outdir / f"{stem}.h"
    depfile = outdir / f"{stem}.d"

    sep = " \\\n  "
    deps = sep.join(sorted(modules))
    contents = f"{c_file} {h_file} : {deps}"

    depfile.write_text(contents)


def get_procs_from_module(user_module):
    symbols = dir(user_module)
    has_export_list = "__all__" in symbols
    if has_export_list:
        exported_symbols = user_module.__dict__["__all__"]
    else:
        exported_symbols = symbols
    library = []
    for sym in exported_symbols:
        if not sym.startswith("_"):
            fn = getattr(user_module, sym)
            if isinstance(fn, exo.Procedure) and not fn.is_instr():
                library.append(fn)
    return library


def load_user_code(path: Path):
    module_path = path.resolve(strict=True)
    module_name = module_path.stem
    module_path = str(module_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    user_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_module)
    return user_module


def main():
    exocc(*sys.argv[1:], name=Path(sys.argv[0]).name)


if __name__ == "__main__":
    main()
