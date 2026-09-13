#!/usr/bin/env python3
"""
Digitize the per-drug FPDc dose-response from Figure 3 of Lee et al. (2025, BBRC 786:152756).

The supplement (mmc1.docx) holds per-drug percentage change in FPD as image-only
tables (Tables S2 to S4), so those values are not machine-readable. Figure 3 of the
main PDF is a 28-row by 4-column heatmap of FPDc at the Cmax ratio doses
(0.1x, 1x, 10x, 100x). This script reads that heatmap.

The figure is a diverging colour map. The legend colours were read from the figure:
  increase (prolongation): #ffd5d5 < #ff9b9b < #ff4f4f < #ff0000
  decrease (shortening):   #d5f4ff < #6dd9ff < #009bd2 < #0070c0
  no effect (< +/-5%):     #d9d9d9
  quiescence:              #ccccff
Each is mapped to a signed ordinal in [-4, +4]. The digitized value is the modal
colour of the cell interior, so it is robust to the printed grid lines.

The output is a coarse ordinal, not the underlying percentage. It is sufficient for a
rank correlation, which uses order only, but it cannot recover magnitudes.

Run: ~/.venvs/myokit/bin/python lee_fig3_extract.py
Output: lee_fpd_ordinals.csv

This digest is reproducible from the published PDF. Every cell's colour is a
plurality, not a clean fill, because a printed concentration number sits inside
each cell. lee_fig3_verify.py re-reads all 112 cells a second way (text pixels
dropped, nearest-palette assignment) and reproduces this file exactly; the
dominant-colour share is 0.72 to 1.00 across cells. Run that script to check.
"""
import os, subprocess, tempfile, csv
from collections import Counter

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "source_data", "1-s2.0-S0006291X2501472X-main.pdf")
OUT = os.path.join(HERE, "lee_fpd_ordinals.csv")

DRUGS = [
    "bepridil", "vandetanib", "quinidine", "dofetilide", "azimilide",
    "disopyramide", "ibutilide", "D,L-sotalol", "chlorpromazine", "clozapine",
    "terfenadine", "risperidone", "cisapride", "astemizole", "clarithromycin",
    "pimozide", "droperidol", "ondansetron", "domperidone", "verapamil",
    "diltiazem", "nitrendipine", "nifedipine", "loratadine", "mexiletine",
    "tamoxifen", "metoprolol", "ranolazine",
]

PALETTE = {
    "#0070c0": -4, "#009bd2": -3, "#6dd9ff": -2, "#d5f4ff": -1,
    "#d9d9d9": 0,
    "#ffd5d5": 1, "#ff9b9b": 2, "#ff4f4f": 3, "#ff0000": 4,
    "#ccccff": None,
}

COL_BANDS = [(1242, 1324), (1327, 1397), (1400, 1470), (1472, 1543)]
ROW_TOP, ROW_BOT = 254, 1032


def hexc(c):
    return "#%02x%02x%02x" % tuple(int(v) for v in c)


def render_page6():
    d = tempfile.mkdtemp(prefix="lee_fig3_")
    subprocess.run(["pdftoppm", "-f", "6", "-l", "6", "-r", "200", "-png",
                    PDF, os.path.join(d, "page")], check=True)
    return os.path.join(d, "page-06.png")


def main():
    img = Image.open(render_page6()).convert("RGB")
    a = np.asarray(img)
    rows = np.linspace(ROW_TOP, ROW_BOT, len(DRUGS) + 1).astype(int)

    def cell(x0, x1, y0, y1):
        blk = a[y0 + 3:y1 - 3, x0 + 3:x1 - 3].reshape(-1, 3)
        return Counter(map(tuple, blk)).most_common(1)[0][0]

    out = []
    for i, drug in enumerate(DRUGS):
        vals = []
        for x0, x1 in COL_BANDS:
            col = hexc(cell(x0, x1, rows[i], rows[i + 1]))
            if col not in PALETTE:
                raise SystemExit(f"unmapped colour {col} for {drug}")
            vals.append(PALETTE[col])
        out.append(dict(drug=drug,
                        cmax_0p1x=vals[0], cmax_1x=vals[1],
                        cmax_10x=vals[2], cmax_100x=vals[3]))

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["drug", "cmax_0p1x", "cmax_1x", "cmax_10x", "cmax_100x"])
        w.writeheader()
        for r in out:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    print(f"wrote {OUT}  ({len(out)} drugs)")


if __name__ == "__main__":
    main()
