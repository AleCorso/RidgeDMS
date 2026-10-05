# Theoretical risk and synthetic-data figures

This folder contains seven self-contained notebooks for the theoretical and
synthetic-data figures. Numerical results are included in `data/`: running the
notebooks with their default `RECOMPUTE=False` redraws and saves the figures
without repeating the expensive risk searches or simulations.

## Setup

From this folder, with Python 3.13:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyter lab notebooks
```

Open a notebook and run its cells in order. Each notebook contains its numerical
settings, plotting function, editable style controls, and PDF export cell.
PDFs are saved in `output/`.

## Figures

| Notebook | Selection | Output PDF |
|---|---|---|
| [01_landscapes](notebooks/01_landscapes.ipynb) | `CASE='set_1'` | `shared_axes_set_1.pdf` |
| same notebook | `CASE='set_2'` | `appendix_landscape_set_2.pdf` |
| same notebook | `CASE='set_6'` | `appendix_landscape_set_6.pdf` |
| [02_library_breadth](notebooks/02_library_breadth.ipynb) | risk cuts and optimal breadth | `library_breadth_with_opt.pdf` |
| [03_reference_effects](notebooks/03_reference_effects.ipynb) | equal-reference contributions | `reference_effects_new.pdf` |
| [04_parameter_sweeps](notebooks/04_parameter_sweeps.ipynb) | `GROUP='matched'` (in-distribution), `'energies'`, or `'means'` | `appendix_sweeps_<GROUP>.pdf` |
| [05_zero_ridge](notebooks/05_zero_ridge.ipynb) | optimal ridge and the slope at zero | `appendix_zero_ridge.pdf` |
| [06_infinite_ridge](notebooks/06_infinite_ridge.ipynb) | finite search caps and infinity certificates | `appendix_infinite_ridge.pdf` |
| [07_finite_size](notebooks/07_finite_size.ipynb) | in-distribution convergence | `appendix_finite_size.pdf` |

Change `CASE` or `GROUP` to reproduce another figure from the same notebook.
Plot settings can be changed without recalculating numerical results. To change
scientific inputs, set `RECOMPUTE=True` and run the numerical stages first.
The bundled-data loader checks that numerical settings match the saved results.
Recomputed results use the local `.cache/` directory; plotting and PDF export
remain separate from computation.

## Data and conventions

The `data/` JSON files contain numerical settings, result structure, and source
hashes; the paired NPZ files contain arrays. The loader verifies the NPZ checksum
and uses `allow_pickle=False`. Synthetic data are generated from the specified
model and seeds. Bundled simulation results contain the summaries needed to
reproduce the figures; training matrices are regenerated only in recompute mode.

Throughout these figures, `R` is excess test risk: the additive independent
test-noise variance is omitted. `R_infinity` uses the same convention. Synthetic
test risk is integrated exactly, and error bars show one standard error over
training-set and noise realizations. The finite-size MSE averages squared
simulation–theory deviations over trials, using theory at the realized `qL/n`.
It includes trial variability and is not the squared error of the trial mean.

Ridge optimization includes the exact zero endpoint. Reaching a finite search
cap does not certify an infinite optimum. Dashed transition connectors use
stored risk-barrier diagnostics; infinity certificates use a separate
sufficient test. The numerical procedures are described in the paper's
numerical-methods appendix.

## Validation

All eleven figure selections were executed from a standalone copy on 5 October
2026 with Python 3.13.11 on Linux x86_64, using the pinned dependencies in `requirements.txt`.
The generated PDFs were visually checked against the manuscript figures. This
check uses the supplied numerical results; it does not repeat the full simulations.

Check Python syntax, notebook structure, local dependencies, and bundled data:

```bash
.venv/bin/python tools/validate_bundle.py
```

Execute every figure selection in a fresh kernel, using the bundled results:

```bash
.venv/bin/python tools/validate_bundle.py --execute
```

This produces all eleven PDFs, a JSON report at
`output/validation_report.json`, and executed notebook copies and PNG previews
in `output/validation/`. Source notebooks are unchanged. To validate one notebook,
add, for example, `--notebook 02_library_breadth`.
