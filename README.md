# Data and code: multi-channel iPSC-CM model versus a hERG-only margin

This deposit accompanies the preprint "A multi-channel iPSC-cardiomyocyte repolarization
model does not beat a hERG-only margin on CiPA risk classification". It holds the analysis
code, the derived result files, the drug reference table, and the manuscript source.

## Contents

- `cipa_validation/` analysis scripts, the 28-drug reference table, and every CSV and JSON
  the scripts produce.
- `ta1_causal_prototype/` the model wrappers and the Kernik-2019 CellML file.
- `preprint/` the manuscript LaTeX source and the figure PDFs.

The layout matches the original working tree, so the scripts run unchanged. Run them from
inside `cipa_validation/`.

## Third-party data are not redistributed

Two inputs are third-party and must be obtained from their sources.

1. Blinova et al. (2018) released per-well Excel data. Save it as
   `cipa_validation/source_data/Blinova_etal_2018_data.xlsx` before running
   `blinova_validation.py`, `margin_comparison.py`, or `collapse_calibration.py`.
2. Lee et al. (2025) main text. Figure 3 is a heatmap of per-drug FPDc change. The
   analysis digitizes that figure; the supplement tables (uncorrected FPD) are image-only
   and are not digitized.

The Kernik-2019 CellML model (`ta1_causal_prototype/kernik_2019.cellml`) comes from the
Physiome Model Repository (exposure e/805) with the Kernik et al. (2019) publication.

## Environment

Python 3 with `numpy`, `scipy`, `openpyxl`, `matplotlib`, and `myokit` built with the
CVODE (SUNDIALS) backend.

## Reproduce

```
cd cipa_validation
python blinova_validation.py        # Blinova rank and direction (uncorrected)
python margin_comparison.py         # multi-channel vs hERG-only margin, three modes
python rate_correction.py           # Fridericia correction, reads margin_comparison_results.csv
python collapse_calibration.py
python i1_mechanism_recovery.py
python i2_calibration.py
python cipa_validation_harness_iks.py
python cipa_validation_harness_inal.py
python cmax_sensitivity.py
python lee_fig3_extract.py          # digitize Lee Figure 3 into ordinals
python lee_fig3_verify.py           # independent second read of the same cells
python lee_comparison.py            # rank the model against the Lee panel
python make_preprint_figures.py     # writes figures into preprint/figures
```

`margin_comparison.py` takes about eight minutes (336 Myokit runs). Run it before
`rate_correction.py`.

## Key result files

- `cipa_validation/margin_comparison_summary.json` rank and direction, three channel sets.
- `cipa_validation/rate_correction_summary.json` the primary corrected Blinova numbers.
- `cipa_validation/direction_agreement_summary.json` direction agreement and the base rate.
- `cipa_validation/cipa_validation_summary.json` the three-class benchmark and the frozen cuts.
- `cipa_validation/margin_comparison_results.csv` per-point APD90, cycle length, and state.

## License

Code is released under the MIT license. The derived result files are released under
CC BY 4.0. The Kernik-2019 CellML file keeps its upstream license.
