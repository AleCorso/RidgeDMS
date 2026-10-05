#!/usr/bin/env python3
"""Check the bundle, optionally redraw every figure in a fresh notebook kernel."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNS = [
    ('01_landscapes', 'CASE', 'set_1', 'shared_axes_set_1.pdf'),
    ('01_landscapes', 'CASE', 'set_2', 'appendix_landscape_set_2.pdf'),
    ('01_landscapes', 'CASE', 'set_6', 'appendix_landscape_set_6.pdf'),
    ('02_library_breadth', None, None, 'library_breadth_with_opt.pdf'),
    ('03_reference_effects', None, None, 'reference_effects_new.pdf'),
    ('04_parameter_sweeps', 'GROUP', 'matched', 'appendix_sweeps_matched.pdf'),
    ('04_parameter_sweeps', 'GROUP', 'energies', 'appendix_sweeps_energies.pdf'),
    ('04_parameter_sweeps', 'GROUP', 'means', 'appendix_sweeps_means.pdf'),
    ('05_zero_ridge', None, None, 'appendix_zero_ridge.pdf'),
    ('06_infinite_ridge', None, None, 'appendix_infinite_ridge.pdf'),
    ('07_finite_size', None, None, 'appendix_finite_size.pdf'),
]


def check_bundle():
    """Validate syntax, local imports, notebook structure, and data checksums."""
    from tools.portable_results import load_results
    errors = []
    modules = {path.stem for path in (ROOT/'dms').glob('*.py')}
    files = sorted((ROOT/'dms').glob('*.py')) + sorted((ROOT/'tools').glob('*.py'))
    notebooks = sorted((ROOT/'notebooks').glob('*.ipynb'))
    cells = 0

    def check_code(source, label, relative_imports=False):
        try:
            tree = ast.parse(source, filename=label)
            compile(tree, label, 'exec')
        except SyntaxError as exc:
            errors.append(str(exc))
            return None
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                name = node.module.split('.')
                local = name[0] if relative_imports and node.level == 1 else None
                if name[0] == 'dms' and len(name) > 1:
                    local = name[1]
                if local and local not in modules:
                    errors.append(f'{label}: missing module {local}')
        return tree

    for path in files:
        check_code(path.read_text(), str(path.relative_to(ROOT)), path.parent.name == 'dms')
    for path in notebooks:
        try:
            nb = nbformat.read(path, as_version=4)
            nbformat.validate(nb)
        except Exception as exc:
            errors.append(f'{path.name}: {exc}')
            continue
        ids = [cell.id for cell in nb.cells]
        if len(set(ids)) != len(ids):
            errors.append(f'{path.name}: repeated cell IDs')
        assigned = {}
        for cell in nb.cells:
            if cell.cell_type != 'code':
                continue
            cells += 1
            if cell.outputs or cell.execution_count is not None:
                errors.append(f'{path.name}/{cell.id}: source notebook contains saved execution')
            tree = check_code(cell.source, f'{path.name}/{cell.id}')
            if tree is not None:
                for node in tree.body:
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                assigned[target.id] = node.value
        if 'PDF_FIGURES' not in assigned:
            errors.append(f'{path.name}: missing PDF_FIGURES mapping')
        recompute = assigned.get('RECOMPUTE')
        if not isinstance(recompute, ast.Constant) or recompute.value is not False:
            errors.append(f'{path.name}: RECOMPUTE must default to False')
    archives = sorted((ROOT/'data').glob('*.json'))
    array_count = 0
    for path in archives:
        try:
            record = json.loads(path.read_text())
            archive = path.with_suffix('.npz')
            if sha256(archive.read_bytes()).hexdigest() != record['npz_sha256']:
                raise ValueError('NPZ checksum mismatch')
            with np.load(archive, allow_pickle=False) as arrays:
                for key in arrays.files:
                    arrays[key]  # Reject object arrays and corrupt payloads.
                array_count += len(arrays.files)
            _, metadata = load_results(path.with_suffix(''), restore_records=True)
            if 'settings' not in metadata:
                raise ValueError('Missing numerical settings')
            for filename, expected in record.get('utility_sha256', {}).items():
                if sha256((ROOT/'dms'/filename).read_bytes()).hexdigest() != expected:
                    raise ValueError(f'utility hash mismatch: {filename}')
        except Exception as exc:
            errors.append(f'{path.name}: {exc}')
    if not archives:
        errors.append('No bundled numerical data found')
    return dict(status='failed' if errors else 'passed', errors=errors,
                notebooks=len(notebooks), code_cells=cells, python_files=len(files),
                data_archives=len(archives), arrays=array_count)


def select_variant(nb, name, value):
    """Replace one selector in an in-memory copy, leaving its source file intact."""
    if name is None:
        return
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        for node in ast.parse(cell.source).body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name):
                lines = cell.source.splitlines(keepends=True)
                lines[node.lineno-1:node.end_lineno] = [f'{name} = {value!r}\n']
                cell.source = ''.join(lines)
                return
    raise ValueError(f'Missing selector {name}')


def execute_run(run, timeout):
    from jupyter_client import KernelManager
    from nbclient import NotebookClient

    notebook, selector, value, filename = run
    tag = notebook + (f'_{value}' if value else '')
    destination = ROOT/'output'/'validation'
    destination.mkdir(parents=True, exist_ok=True)
    path = ROOT/'notebooks'/f'{notebook}.ipynb'
    nb = nbformat.read(path, as_version=4)
    select_variant(nb, selector, value)
    preview = destination/f'{tag}.png'
    metrics = destination/f'{tag}_figure.json'
    nb.cells.append(nbformat.v4.new_code_cell(f'''
# Render validation for the selected exported figure.
_validation_name = next(key for key, name in PDF_FIGURES.items() if name == {filename!r})
_validation_registries = [globals().get(name, {{}}) for name in ('FIGURES', 'APPENDIX_FIGURES', 'figures')]
_validation_figure = next(registry[_validation_name] for registry in _validation_registries if _validation_name in registry)
_validation_figure.canvas.draw()
assert len(_validation_figure.axes) > 0, 'Figure has no axes'
assert all(ax.get_position().width > 0.001 and ax.get_position().height > 0.001 for ax in _validation_figure.axes), 'Collapsed figure axes'
_validation_figure.savefig({str(preview)!r}, dpi=150, bbox_inches='tight')
import json as _validation_json
from pathlib import Path as _ValidationPath
_ValidationPath({str(metrics)!r}).write_text(_validation_json.dumps({{
    'axes': len(_validation_figure.axes),
    'size_inches': _validation_figure.get_size_inches().tolist(),
    'lines': sum(len(ax.lines) for ax in _validation_figure.axes),
    'collections': sum(len(ax.collections) for ax in _validation_figure.axes),
    'images': sum(len(ax.images) for ax in _validation_figure.axes),
}}, indent=2) + '\\n')
'''))
    pdf = ROOT/'output'/filename
    previous = pdf.stat().st_mtime_ns if pdf.exists() else None
    started = perf_counter()
    result = dict(notebook=path.name, selector={selector: value} if selector else {},
                  pdf=f'output/{filename}', status='failed')
    try:
        # Use the Python interpreter running this validator, irrespective of any
        # other Jupyter environments registered on the machine.
        manager = KernelManager(kernel_name='python3')
        manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        client = NotebookClient(nb, km=manager, timeout=timeout,
                                resources={'metadata': {'path': str(ROOT)}},
                                allow_errors=False)
        client.execute(cleanup_kc=True)
        if not pdf.is_file() or pdf.stat().st_mtime_ns == previous:
            raise ValueError('Expected PDF was not regenerated')
        if pdf.stat().st_size < 1000 or not pdf.read_bytes().startswith(b'%PDF-'):
            raise ValueError('Expected output is not a nonempty PDF')
        result.update(status='passed', pdf_bytes=pdf.stat().st_size,
                      pdf_sha256=sha256(pdf.read_bytes()).hexdigest(),
                      figure=json.loads(metrics.read_text()),
                      preview=str(preview.relative_to(ROOT)))
    except Exception as exc:
        result['error'] = str(exc)
    finally:
        result['elapsed_seconds'] = round(perf_counter()-started, 3)
        nbformat.write(nb, destination/f'{tag}.ipynb')
    return result


def environment():
    packages = {}
    for name in ('numpy', 'matplotlib', 'scipy', 'ipython', 'jupyterlab',
                 'ipykernel', 'jupyter_client', 'nbclient', 'nbformat'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(python=platform.python_version(), platform=platform.system(),
                architecture=platform.machine(), packages=packages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Redraw all selected figures using bundled data')
    parser.add_argument('--notebook', action='append', choices=sorted({run[0] for run in RUNS}),
                        help='Restrict execution to this notebook; can be repeated')
    parser.add_argument('--timeout', type=int, default=600, help='Maximum seconds per notebook cell')
    args = parser.parse_args()
    report = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  environment=environment(), static=check_bundle(), runs=[])
    destination = ROOT/'output'/'validation_report.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if args.execute and report['static']['status'] == 'passed':
        for run in RUNS:
            if args.notebook and run[0] not in args.notebook:
                continue
            print(f'Running {run[0]} {run[2] or ""}', flush=True)
            result = execute_run(run, args.timeout)
            report['runs'].append(result)
            destination.write_text(json.dumps(report, indent=2)+'\n')
            print(f'  {result["status"]}: {result["pdf"]}', flush=True)
    report['status'] = ('passed' if report['static']['status'] == 'passed'
                        and all(run['status'] == 'passed' for run in report['runs']) else 'failed')
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
