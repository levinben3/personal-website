#!/usr/bin/env python3
"""Group GalDisc 4XMM-DR14s sources by absorption, extinction, flux and hardness ratio.

Script version of stacking.ipynb, with the notebook's logic errors corrected (see grouping_common.py).

    python group_sources_hardness.py                          # 2x20 deg nH-map region
    python group_sources_hardness.py --coverage extended      # also group sources outside the 2x20 nH map
    python group_sources_hardness.py --stack                  # group, then stack with combine.py

Outputs (in --out): groups.json, group_lists/group_NNNN.txt (catalog indices, the
combine.py input format), source_status.csv (what happened to every catalog row),
catalog_with_groups.fits (catalog + GROUP_ID column).
"""

import argparse

import grouping_common as gc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    gc.add_common_args(p)
    p.add_argument('--hardness-column', default='EPhr2', choices=['EPhr2', 'EP_HR4'],
                   help="EPhr2 (default): (4.5-12 - 2-4.5 keV)/(sum) from summed PN+MOS count rates, computed "
                        "in the notebook; EP_HR4: the catalog column the notebook grouped on, NaN for 96%% of sources")
    opts = p.parse_args()
    opts.out = opts.out or f'grouping_hardness_{opts.hardness_column}_{opts.coverage}'

    def shape_values_for(cat):
        return {int(s): h for s, h in zip(cat['src_id'], cat[opts.hardness_column])}

    def make_binner(values):
        return gc.hardness_binner(), {'labels': gc.HARDNESS_LABELS, 'special': ['1+ (==1)', '0_exact (==0)']}

    gc.run_grouping(opts, 'hardness', shape_values_for, make_binner, lambda v: gc.HARDNESS_TOL,
                    extra_config={'hardness_column': opts.hardness_column})


if __name__ == '__main__':
    main()
