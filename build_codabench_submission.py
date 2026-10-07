from __future__ import annotations

"""
Build the final Codabench Track-2 code-submission ZIP.

The ZIP root is guaranteed to contain:
  run.py
  metadata.yaml
  abpr_runtime.py
  assets/model.pt

It also refuses to package the organizer's prior-only baseline run.py.
"""

from pathlib import Path
import argparse
import ast
import json
import shutil
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--runtime-zip',
        default='/content/drive/MyDrive/PedestrianAttributeRecognition/ABPR_Results/UPAR2027_Track2_ABPR_Runtime.zip',
    )
    p.add_argument(
        '--result-dir',
        default='/content/drive/MyDrive/PedestrianAttributeRecognition/ABPR_Results',
    )
    p.add_argument('--name', default='Submission_ABPR_SCORE_BOOST')
    args = p.parse_args()

    runtime_zip = Path(args.runtime_zip)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    if not runtime_zip.exists():
        raise FileNotFoundError(runtime_zip)

    folder = result_dir / args.name
    final_zip = result_dir / f'{args.name}.zip'
    if folder.exists():
        shutil.rmtree(folder)
    if final_zip.exists():
        final_zip.unlink()
    folder.mkdir(parents=True)

    with zipfile.ZipFile(runtime_zip, 'r') as zf:
        zf.extractall(folder)

    required = [
        folder / 'run.py',
        folder / 'metadata.yaml',
        folder / 'abpr_runtime.py',
        folder / 'assets' / 'model.pt',
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError(f'Runtime package is missing required files: {missing}')

    run_text = (folder / 'run.py').read_text(encoding='utf-8')
    tree = ast.parse(run_text)
    funcs = [
        n.name for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if 'rank_gallery' not in funcs:
        raise RuntimeError('run.py does not define rank_gallery().')
    if 'ABPRRuntime' not in run_text or 'model.pt' not in run_text:
        raise RuntimeError(
            'run.py is not connected to the trained model. Refusing to build ZIP.'
        )
    forbidden = ['attribute_prior.json', '_PRIOR']
    found = [x for x in forbidden if x in run_text]
    if found:
        raise RuntimeError(
            f'Prior-only organizer baseline detected in final run.py: {found}'
        )

    # Remove any stale sample-baseline assets if present.
    stale = folder / 'assets' / 'attribute_prior.json'
    if stale.exists():
        stale.unlink()

    # ZIP CONTENTS, not the wrapper folder.
    with zipfile.ZipFile(
        final_zip,
        'w',
        zipfile.ZIP_DEFLATED,
        compresslevel=1,
        allowZip64=True,
    ) as zf:
        for f in sorted(folder.rglob('*')):
            if f.is_file():
                zf.write(f, f.relative_to(folder).as_posix())

    with zipfile.ZipFile(final_zip, 'r') as zf:
        names = [n for n in zf.namelist() if not n.endswith('/')]

    for required_name in (
        'run.py',
        'metadata.yaml',
        'abpr_runtime.py',
        'assets/model.pt',
    ):
        if required_name not in names:
            raise RuntimeError(f'{required_name} is not at the expected ZIP path.')

    if any(n.startswith(folder.name + '/') for n in names):
        raise RuntimeError('The wrapper folder was accidentally included in the ZIP.')

    manifest = {
        'submission_zip': str(final_zip),
        'runtime_zip': str(runtime_zip),
        'run_functions': funcs,
        'zip_files': names,
        'critical_checks': {
            'run_py_at_root': True,
            'run_py_uses_ABPRRuntime': True,
            'run_py_uses_model_pt': True,
            'organizer_prior_baseline_removed': True,
        },
    }
    (result_dir / f'{args.name}_manifest.json').write_text(
        json.dumps(manifest, indent=2),
        encoding='utf-8',
    )

    print(json.dumps(manifest, indent=2))
    print('\nUPLOAD THIS FILE DIRECTLY:')
    print(final_zip)
    print(f'Size: {final_zip.stat().st_size / 1024**2:.2f} MiB')


if __name__ == '__main__':
    main()
