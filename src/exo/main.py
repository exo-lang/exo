import argparse
import importlib
import importlib.machinery
import importlib.util
import sys
import textwrap
from pathlib import Path

import exo


def main():
    parser = make_argument_parser()
    args = parser.parse_args()

    library = load_libraries(args.source)

    try:
        if args.debug:
            debug_mode(args, library)
        else:
            exo.compile_procs_to_files(library, *resolve_paths(parser, args))

    except exo.SchedulingError as e:
        sys.exit(str(e))

    except AssertionError as e:
        sys.exit(
            textwrap.dedent(
                f"""
                **INTERNAL COMPILER ERROR**
                {e}
                
                Please file a report at https://github.com/exo-lang/exo/issues
                """
            )
        )


def load_libraries(source_files):
    library = [
        proc
        for mod in source_files
        for proc in get_procs_from_module(load_user_code(mod))
    ]
    return library


def make_argument_parser():
    parser = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name, description="Compile an Exo library."
    )

    simple = parser.add_argument_group()
    simple.add_argument(
        "-o",
        "--outdir",
        metavar="OUTDIR",
        help=(
            "output directory for .c and .h files. Mutually exclusive with "
            "--cfile/--hfile."
        ),
    )
    simple.add_argument(
        "-s",
        "--stem",
        help=(
            "base name for .c and .h files. Defaults to the stem of the single Exo "
            "source file being processed. Required when compiling multiple source "
            "files."
        ),
    )

    full_paths = parser.add_argument_group()
    full_paths.add_argument(
        "-oc",
        "--cfile",
        metavar="CFILE",
        help="output path for .c file. Mutually exclusive with --outdir/--stem.",
    )
    full_paths.add_argument(
        "-oh",
        "--hfile",
        metavar="HFILE",
        help="output path for .h file Mutually exclusive with --outdir/--stem.",
    )

    parser.add_argument("source", type=str, nargs="+", help="source files to compile")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s version {exo.__version__}",
        help="print the version and exit",
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str,
        metavar="DEBUG",
        choices=["schedule"],
        help=(
            "Run the specified debugging tool. 'schedule' prints the fully scheduled "
            "procs prior to codegen"
        ),
    )
    return parser


def debug_mode(args, library):
    if args.debug == "schedule":
        for p in library:
            print(p)
    else:
        assert False, f"Unknown debug mode `{args.debug}`"


def resolve_paths(parser, args):
    if args.outdir and (args.cfile or args.hfile):
        parser.error(
            "The --outdir/--stem output mode is mutually exclusive with the "
            "--cfile/--hfile output mode"
        )

    if not args.outdir and not args.cfile and not args.hfile:
        parser.error(
            "Must specify output either by providing an --outdir and a --stem (when "
            "processing multiple source files) OR by providing the --cfile and --hfile "
            "output paths."
        )

    if args.outdir:
        if args.stem:
            stem = args.stem
        elif len(args.source) == 1:
            stem = Path(args.source[0]).stem
        else:
            parser.error("Must provide --stem when processing multiple source files.")

        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        outdir = outdir.resolve(strict=True)
        return outdir / f"{stem}.c", outdir / f"{stem}.h"

    else:
        if not args.cfile:
            parser.error("Must pass --cfile/--hfile when not using --outdir/--stem")
        cfile = Path(args.cfile)
        hfile = Path(args.hfile)

        cfile.parent.mkdir(parents=True, exist_ok=True)
        hfile.parent.mkdir(parents=True, exist_ok=True)

        return cfile, hfile


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
