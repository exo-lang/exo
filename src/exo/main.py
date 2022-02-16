import argparse
import importlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import exo


def main():
    parser = argparse.ArgumentParser(prog=Path(sys.argv[0]).name,
                                     description='Compile an Exo library.')
    parser.add_argument('-o', '--outdir', metavar='OUTDIR', required=True,
                        help='output directory for build artifacts')
    parser.add_argument('--stem', required=True,
                        help='base name for .c and .h files')
    parser.add_argument('source', type=str, nargs='+',
                        help='source file to compile')
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s version {exo.__version__}',
                        help='print the version and exit')

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outdir = outdir.resolve(strict=True)

    library = [
        proc
        for mod in args.source
        for proc in get_procs_from_module(load_user_code(mod))
    ]

    exo.compile_procs(library, outdir, f'{args.stem}.c', f'{args.stem}.h')


def get_procs_from_module(user_module):
    symbols = dir(user_module)
    has_export_list = '__all__' in symbols
    if has_export_list:
        exported_symbols = user_module.__dict__['__all__']
    else:
        exported_symbols = symbols
    library = []
    for sym in exported_symbols:
        if not sym.startswith('_'):
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


if __name__ == '__main__':
    main()
