import argparse
import importlib
import importlib.machinery
import importlib.util
import inspect
import sys

sys.setrecursionlimit(10000)

from pathlib import Path

import exo


def main():
    parser = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name, description="Compile an Exo library."
    )
    parser.add_argument(
        "-o",
        "--outdir",
        metavar="OUTDIR",
        help="output directory for build artifacts",
    )
    parser.add_argument("--stem", help="base name for .c and .h files")
    parser.add_argument("source", type=str, nargs="+", help="source file to compile")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s version {exo.__version__}",
        help="print the version and exit",
    )

    args = parser.parse_args()
    srcname = Path(args.source[0]).stem

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
            stem = srcname
        else:
            parser.error("Must provide --stem when processing multiple source files.")
    else:
        stem = args.stem

    library = [
        proc
        for mod in args.source
        for proc in get_procs_from_module(load_user_code(mod))
    ]

    exo.compile_procs(library, outdir, f"{stem}.c", f"{stem}.h")
    write_depfile(outdir, stem)


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


def load_user_code(path):
    module_path = Path(path).resolve(strict=True)
    module_name = module_path.stem
    module_path = str(module_path)
    loader = importlib.machinery.SourceFileLoader(module_name, module_path)
    spec = importlib.util.spec_from_loader(module_name, loader)
    user_module = importlib.util.module_from_spec(spec)
    loader.exec_module(user_module)
    return user_module


if __name__ == "__main__":
    main()
