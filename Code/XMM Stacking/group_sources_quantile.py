#!/usr/bin/env python3
"""Group GalDisc 4XMM-DR14s sources by absorption, extinction, flux and an energy quantile.

Same pipeline as group_sources_hardness.py, with the hardness ratio replaced by a
normalised energy quantile from quantiles.ipynb / compute_quantiles.py
(Q = (E_x% - 0.2 keV) / (12 - 0.2 keV), background subtracted).

Binning: the hardness pipeline uses 4 bins over HR in [-1, 1] and a merge/clip tolerance of
one bin width (0.5). By default the quantile is binned into 4 equal-population bins (upper
edges at the 25th/50th/75th percentiles of the sources being grouped, last edge 1.0), and the
tolerance is the half inter-quartile range, i.e. about one bin width. Both can be overridden.

    python group_sources_quantile.py                                 # median (q50) of quantile_net
    python group_sources_quantile.py --quantile q25 --edges 0.1,0.2,0.3,1 --tol 0.1
    python group_sources_quantile.py --coverage extended --stack
"""

import argparse
import json

import numpy as np

import grouping_common as gc

QUANTILES_JSON = '/home/benjamin/spectra/quantiles.json'
QUANTILE_NAMES = ['q25', 'q33', 'q50', 'q67', 'q75']   # order used in quantiles.json


def load_quantiles(path, field, name):
    """{src_id: value}; empty entries and the notebook's 0 'too few counts' sentinel become NaN."""
    k = QUANTILE_NAMES.index(name)
    with open(path) as f:
        rows = json.load(f)
    out = {}
    for row in rows:
        v = row[field][0][k] if row[field] else float('nan')
        out[int(row['source_id'])] = float('nan') if v == 0 else float(v)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    gc.add_common_args(p)
    p.add_argument('--quantiles-json', default=QUANTILES_JSON)
    p.add_argument('--quantile', default='q50', choices=QUANTILE_NAMES)
    p.add_argument('--quantile-field', default='quantile_net', choices=['quantile_net', 'quantile'],
                   help='quantile_net: interpolated, background-subtracted (used for the error bars and '
                        'colour-colour plots); quantile: simple discrete estimate')
    p.add_argument('--edges', help='comma-separated upper bin edges (default: 25/50/75th percentiles, 1.0)')
    p.add_argument('--tol', type=float, help='merge/clip tolerance (default: half the inter-quartile range)')
    opts = p.parse_args()
    opts.out = opts.out or f'grouping_quantile_{opts.quantile}_{opts.coverage}'

    quantiles = load_quantiles(opts.quantiles_json, opts.quantile_field, opts.quantile)

    def shape_values_for(cat):
        return {int(s): quantiles.get(int(s), float('nan')) for s in cat['src_id']}

    def make_binner(values):
        if opts.edges:
            edges = [float(e) for e in opts.edges.split(',')]
        else:
            finite = values[np.isfinite(values)]
            edges = [float(e) for e in np.percentile(finite, [25, 50, 75])] + [1.0]
        return gc.edge_binner(edges), {'upper_edges': edges}

    def tol_for(values):
        if opts.tol is not None:
            return opts.tol
        q25, q75 = np.percentile(values[np.isfinite(values)], [25, 75])
        return float((q75 - q25) / 2)

    gc.run_grouping(opts, opts.quantile, shape_values_for, make_binner, tol_for,
                    extra_config={'quantiles_json': opts.quantiles_json, 'quantile_field': opts.quantile_field})


if __name__ == '__main__':
    main()
