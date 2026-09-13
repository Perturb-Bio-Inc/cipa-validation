#!/usr/bin/env python3
"""
Independent check of lee_fpd_ordinals.csv.

lee_fig3_extract.py reads the modal colour of each Figure 3 cell. A printed
concentration number sits inside every cell, so that modal colour is only a
plurality (17-87% of the cell pixels), not a clean fill. This script re-reads the
same cells a second, independent way and confirms the result.

Difference from lee_fig3_extract.py:
  - drops dark pixels (the printed number) before reading the colour
  - assigns every remaining pixel to the nearest of the ten legend colours,
    instead of requiring an exact modal match
  - uses a wider 300 dpi render and different row boundaries
  - reports the dominant-colour share, so an ambiguous cell would show up

If any cell disagrees with lee_fpd_ordinals.csv, this script exits non-zero.

Run: ~/.venvs/myokit/bin/python lee_fig3_verify.py
"""
import os, sys, csv, subprocess, tempfile
from collections import Counter

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "source_data", "1-s2.0-S0006291X2501472X-main.pdf")
ORD = os.path.join(HERE, "lee_fpd_ordinals.csv")

DRUGS = [
    "bepridil", "vandetanib", "quinidine", "dofetilide", "azimilide",
    "disopyramide", "ibutilide", "D,L-sotalol", "chlorpromazine", "clozapine",
    "terfenadine", "risperidone", "cisapride", "astemizole", "clarithromycin",
    "pimozide", "droperidol", "ondansetron", "domperidone", "verapamil",
    "diltiazem", "nitrendipine", "nifedipine", "loratadine", "mexiletine",
    "tamoxifen", "metoprolol", "ranolazine",
]

PALETTE = [
    ((0x00, 0x70, 0xc0), -4), ((0x00, 0x9b, 0xd2), -3),
    ((0x6d, 0xd9, 0xff), -2), ((0xd5, 0xf4, 0xff), -1),
    ((0xd9, 0xd9, 0xd9), 0),
    ((0xff, 0xd5, 0xd5), 1), ((0xff, 0x9b, 0x9b), 2),
    ((0xff, 0x4f, 0x4f), 3), ((0xff, 0x00, 0x00), 4),
    ((0xcc, 0xcc, 0xff), None),
]

# Right panel (Cmax ratio) cell columns, 300 dpi. Each holds the printed
# concentration 0.1x / 1x / 10x / 100x of the row's Cmax.
COLS = [(1884, 1986), (2011, 2096), (2127, 2205), (2236, 2314)]
ROW_TOP, ROW_BOT, DPI = 380, 1540, 300

FIELDS = ["cmax_0p1x", "cmax_1x", "cmax_10x", "cmax_100x"]


def render():
    d = tempfile.mkdtemp(prefix="lee_fig3_verify_")
    subprocess.run(["pdftoppm", "-f", "6", "-l", "6", "-r", str(DPI), "-png",
                    PDF, os.path.join(d, "page")], check=True)
    return os.path.join(d, "page-06.png")


def read_cell(arr, x0, x1, y0, y1):
    blk = arr[y0:y1, x0:x1].reshape(-1, 3)
    keep = blk[blk.max(axis=1) > 110]
    labels = []
    for px in keep:
        label = min(PALETTE, key=lambda po: sum(
            (int(px[k]) - po[0][k]) ** 2 for k in range(3)))[1]
        labels.append(label)
    ordinal, n = Counter(labels).most_common(1)[0]
    return ordinal, n / len(labels)


def main():
    arr = np.asarray(Image.open(render()).convert("RGB")).astype(int)
    rows = np.linspace(ROW_TOP, ROW_BOT, len(DRUGS) + 1).astype(int)

    committed = {}
    with open(ORD) as fh:
        for r in csv.DictReader(fh):
            committed[r["drug"]] = [None if r[k] == "" else int(r[k]) for k in FIELDS]

    ok = True
    for i, drug in enumerate(DRUGS):
        got, shares = [], []
        for x0, x1 in COLS:
            o, share = read_cell(arr, x0, x1, rows[i] + 4, rows[i + 1] - 4)
            got.append(o)
            shares.append(share)
        match = got == committed[drug]
        ok = ok and match
        cells = " ".join(f"{o}:{s:.2f}" for o, s in zip(got, shares))
        print(f"{drug:15} {cells:34} committed={committed[drug]} {'ok' if match else 'MISMATCH'}")

    if not ok:
        sys.exit("verification FAILED: a cell disagrees with lee_fpd_ordinals.csv")
    print(f"\nall {len(DRUGS)} rows and {len(DRUGS)*4} cells match lee_fpd_ordinals.csv")


if __name__ == "__main__":
    main()
