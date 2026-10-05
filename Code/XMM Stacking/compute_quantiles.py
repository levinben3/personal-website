#!/usr/bin/env python3
"""Per-source energy quantiles (script port of quantiles.ipynb); writes quantiles.json.

For every catalog source, reads the combined per-source spectrum in
{spec_dir}/{catalog_index}_spec/ (*_spectrum.fits, *_background.fits, *_response.fits;
PN preferred, then MOS1, then MOS2), background-subtracts, and computes the normalised
quantiles Q_x = (E_x - E_lo) / (E_up - E_lo) for x = 25, 33, 50, 67, 75 over 0.2-12 keV.

Output format is the notebook's: a list of
{'source_id', 'quantile': [[5 values]], 'quantile_net': [[5 values]], 'error_net': [[5 values]]},
with empty lists for sources whose files are missing or have BACKSCAL problems.

    python compute_quantiles.py --out quantiles.json [--spec-dir /data/tong/xmm/allspec]
"""

import argparse
import glob
import json
import math
import os

import numpy as np
from astropy.io import fits
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import gammaln

import grouping_common as gc

SPEC_DIR = '/data/tong/xmm/allspec'
E_MIN, E_MAX = 0.2, 12.0
PERCENTS = [25, 33, 50, 67, 75]
BAD = ([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [], [0, 0, 0, 0, 0], True)


def _pick(files, first='pn', fallbacks=('mos1', 'mos2')):
    """First PN file, else the first file that mentions MOS1 or MOS2 (notebook order)."""
    for f in files:
        if first in f.lower():
            return f
    for f in files:
        fl = f.lower()
        if any(k in fl for k in fallbacks):
            return f
    return None


def quantile_net_compute(net_energies_sorted, x, a_scal, N):
    energy_grid = np.unique(net_energies_sorted)
    IC_above = N - np.searchsorted(net_energies_sorted, energy_grid, side='right')
    IC_below = np.searchsorted(net_energies_sorted, energy_grid, side='left')
    x_list = (N + IC_above - IC_below) / (2 * N)
    x_unique, idx = np.unique(x_list, return_index=True)
    energy_unique = energy_grid[idx]
    interp_func = interp1d(x_unique[::-1], energy_unique[::-1], bounds_error=False, fill_value="extrapolate")
    return interp_func(1 - (x / 100))


def error_bars(sorted_energies, x, src_counts, bkg_counts, E_lo, E_up):
    """Maritz-Jarrett style quantile uncertainty, as in the notebook.

    NOTE (preserved): x is converted to a fraction before the `x == 50` / `x == 33` tests,
    so the minimum-count threshold is always 6.
    """
    x = x / 100
    N = len(sorted_energies)
    M = N * x + 0.5
    a = M - 1
    b = N - M
    C1 = 0
    C2 = 0
    factor = math.sqrt(1 + (np.sum(bkg_counts) / np.sum(src_counts)))

    def integrand(t, a, b):
        return t ** (a - 1) * ((1 - t) ** (b - 1))

    def beta_prefactor(a, b):
        return np.exp(gammaln(a + b) - gammaln(a) - gammaln(b))

    def B(a, b, i):
        return beta_prefactor(a, b) * quad(integrand, 0, i / N, args=(a, b))[0]

    if x == 50:
        minimum = 3
    elif x == 33 or x == 67:
        minimum = 5
    else:
        minimum = 6
    if len(sorted_energies) < minimum:
        return 0
    for i, photon in enumerate(sorted_energies):
        W = B(a, b, i + 1) - B(a, b, i)
        C1 = C1 + W * photon
        C2 = C2 + W * (photon ** 2)
    variance = C2 - C1 ** 2
    if not variance > 0:          # also catches NaN, as the notebook's `if variance > 0` did
        return 0
    sigma_Q = factor * math.sqrt(variance) if N < 30 else math.sqrt(variance)
    return sigma_Q / (E_up - E_lo)


def quantile_compute(min_energy, max_energy, spec_path):
    """Quantiles for one source directory. Returns (quantile, quantile_net, error, error_net, bad)."""
    spec_file = _pick(glob.glob(os.path.join(spec_path, '*_spectrum.fits')))
    if spec_file is None:
        return BAD
    bkg_file = _pick(glob.glob(os.path.join(spec_path, '*_background.fits')))
    if bkg_file is None:
        return BAD
    rmf_file = _pick(glob.glob(os.path.join(spec_path, '*_response.fits')))
    if rmf_file is None:
        return BAD

    with fits.open(spec_file) as hdul:
        src_counts = hdul[1].data['COUNTS']
        src_backscal = hdul[1].header.get('BACKSCAL')
    with fits.open(bkg_file) as hdul:
        bkg_counts = hdul[1].data['COUNTS']
        bkg_backscal = hdul[1].header.get('BACKSCAL')
    with fits.open(rmf_file) as hdul:
        e_min = hdul['EBOUNDS'].data['E_MIN']
        e_max = hdul['EBOUNDS'].data['E_MAX']

    if src_backscal is None or bkg_backscal is None or bkg_backscal == 0 or src_backscal == 0:
        return BAD

    E_lo, E_up = min_energy, max_energy
    a_scal = src_backscal / bkg_backscal
    energy_midpoints = (e_min + e_max) / 2
    energy_mask = (energy_midpoints >= E_lo) & (energy_midpoints <= E_up)
    energy_midpoints_cut = energy_midpoints[energy_mask]
    src_counts = src_counts[energy_mask]
    bkg_counts = bkg_counts[energy_mask]
    net_counts = np.floor(src_counts - bkg_counts * a_scal).astype(int)
    net_counts[net_counts < 0] = 0
    net_energies_sorted = np.sort(np.repeat(energy_midpoints_cut, net_counts))

    N = len(net_energies_sorted)
    quantile, error, quantile_net, error_net = [], [], [], []
    for percent in PERCENTS:
        minimum = 2 if percent in (33, 50, 67) else 3
        if N <= minimum:
            quantile.append(0)
            quantile_net.append(0)
            error_net.append(0)
            continue
        quantile_net.append((quantile_net_compute(net_energies_sorted, percent, a_scal, N) - E_lo) / (E_up - E_lo))
        error_net.append(error_bars(net_energies_sorted, percent, src_counts, bkg_counts, E_lo, E_up))
        # NOTE (preserved): i = x*N + 0.5 is used directly as a 0-based index.
        i_low = int(np.floor(((percent / 100) * (2 * N) + 1) / 2))
        if i_low + 1 > N - 1:
            energy = net_energies_sorted[i_low]
        else:
            energy = (net_energies_sorted[i_low] + net_energies_sorted[i_low + 1]) / 2
        quantile.append((energy - E_lo) / (E_up - E_lo))
    return quantile, quantile_net, error, error_net, False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--catalog', default=gc.CATALOG)
    p.add_argument('--spec-dir', default=SPEC_DIR)
    p.add_argument('--out', default='quantiles.json')
    opts = p.parse_args()

    with fits.open(opts.catalog) as hdul:
        src_ids = np.array(hdul[1].data['SRCID'])
    rows, n_bad = [], 0
    for index, src_id in enumerate(src_ids):
        q, qn, _, en, bad = quantile_compute(E_MIN, E_MAX, os.path.join(opts.spec_dir, f'{index}_spec'))
        row = {'source_id': int(src_id), 'quantile': [], 'quantile_net': [], 'error_net': []}
        if bad:
            n_bad += 1
        else:
            row['quantile'].append([float(v) for v in q])
            row['quantile_net'].append([float(v) for v in qn])
            row['error_net'].append([float(v) for v in en])
        rows.append(row)
        if (index + 1) % 500 == 0:
            print(f'{index + 1}/{len(src_ids)} sources, {n_bad} without usable spectra')
    with open(opts.out, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f'wrote {opts.out}: {len(rows)} sources, {n_bad} without usable spectra')


if __name__ == '__main__':
    main()
