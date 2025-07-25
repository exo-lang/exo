import argparse
import importlib
import importlib.util
import inspect
import sys

from pathlib import Path

import exo


def main(*args, name="exocc"):
    sys.setrecursionlimit(10000)

    parser = argparse.ArgumentParser(prog=name, description="Compile an Exo library.")
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

    args = parser.parse_args(args)
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


def _discover_package_parts(start_dir: Path):
    parts = []
    current = start_dir.resolve()
    while (current / "__init__.py").exists():
        parts.append(current.name)
        current = current.parent
    parts.reverse()
    return str(current), parts


def load_user_code(path):
    module_path = Path(path).resolve(strict=True)

    if module_path.name == "__init__.py":
        raise ValueError(
            "Do not pass __init__.py directly. Pass the package directory instead."
        )

    if module_path.is_dir():
        file_path = module_path / "__init__.py"
        if not file_path.exists():
            raise ValueError(
                f"Directory '{module_path}' is not a package (missing __init__.py)"
            )
        stem = None
        base_dir = module_path
    else:
        file_path = module_path
        stem = module_path.stem
        base_dir = module_path.parent

    package_root, pkg_parts = _discover_package_parts(base_dir)
    package_name = ".".join(pkg_parts) if pkg_parts else None
    module_name = (
        ".".join(pkg_parts + [stem]) if stem else package_name or module_path.stem
    )

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package_name

    sys.path.insert(0, package_root)
    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


if __name__ == "__main__":
    main(*sys.argv[1:], name=Path(sys.argv[0]).name)
