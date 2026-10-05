#!/usr/bin/env python3
"""Stack the spectra of each source group with stacking_scripts/combine.py.

Reads groups.json from group_sources_hardness.py / group_sources_quantile.py and, for every
group with at least --min-size sources and every detector, collects each member's
spectrum/background/RMF/ARF sets with combine.find_source_files() and merges them with
combine.combine_spectra() (SAS epicspeccombine). combine.py is imported unchanged;
only its BASE_DIR is pointed at --spec-dir.

Needs the SAS environment (e.g. `setxmm22`) unless --dry-run is given.

    python stack_groups.py grouping_hardness_EP_HR4_footprint/groups.json --dry-run
    python stack_groups.py grouping_hardness_EP_HR4_footprint/groups.json --groups 3,7
"""

import argparse
import json
import os
import shutil
import sys

STACKING_SCRIPTS = '/home/benjamin/stacking_scripts'
SPEC_DIR = '/data/tong/xmm/allspec'          # {catalog_index}_spec/ per source, as combine.BASE_DIR
DETECTORS = ['mos1', 'mos2', 'pn']

sys.path.insert(0, STACKING_SCRIPTS)
import combine  # noqa: E402


def stack_from_groups_file(groups_file, spec_dir=None, out_dir=None, min_size=2, dry_run=False,
                           group_ids=None, detectors=DETECTORS):
    spec_dir = spec_dir or SPEC_DIR
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(groups_file)), 'stacked')
    if not dry_run and shutil.which('epicspeccombine') is None:
        raise SystemExit('epicspeccombine not found: initialise SAS first (e.g. `setxmm22`) or use --dry-run')
    combine.BASE_DIR = spec_dir
    os.makedirs(out_dir, exist_ok=True)

    with open(groups_file) as f:
        groups = json.load(f)['groups']
    todo = [g for g in groups if g['n_sources'] >= min_size and (group_ids is None or g['group_id'] in group_ids)]
    print(f'{len(todo)} groups to stack from {groups_file} (spectra in {spec_dir}, output in {out_dir})')

    report = []
    for g in todo:
        gid = g['group_id']
        indices = [s['catalog_index'] for s in g['sources']]
        with open(os.path.join(out_dir, f'group_{gid:04d}_sources.txt'), 'w') as f:
            f.write('\n'.join(str(i) for i in indices) + '\n')
        log_file = os.path.join(out_dir, f'group_{gid:04d}.log')
        for det in detectors:
            files, with_spectra = [], 0
            for idx in indices:
                found = combine.find_source_files(idx, det)
                if found:
                    with_spectra += 1
                    files.extend(found)
            entry = {'group_id': gid, 'detector': det, 'n_sources': len(indices),
                     'n_sources_with_spectra': with_spectra, 'n_file_sets': len(files)}
            if not files:
                entry['result'] = 'no valid spectra'
            elif dry_run:
                entry['result'] = 'dry run'
                entry['spectra'] = [s['spectrum'] for s in files]
            else:
                pha, bkg, rsp = combine.combine_spectra(files, out_dir, log_file, outname=f'group{gid:04d}_{det}')
                entry['result'] = 'ok' if pha else 'epicspeccombine failed (see log)'
                entry.update({'spectrum': pha, 'background': bkg, 'response': rsp})
            print(f"group {gid:4d} {det:4s}: {with_spectra}/{len(indices)} sources with spectra, "
                  f"{len(files)} file sets -> {entry['result']}")
            report.append(entry)

    with open(os.path.join(out_dir, 'stack_report.json'), 'w') as f:
        json.dump(report, f, indent=1)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('groups_file')
    p.add_argument('--spec-dir', default=SPEC_DIR)
    p.add_argument('--out-dir')
    p.add_argument('--min-size', type=int, default=2)
    p.add_argument('--groups', help='comma-separated group ids to stack (default: all)')
    p.add_argument('--detectors', default=','.join(DETECTORS))
    p.add_argument('--dry-run', action='store_true', help='list matched files without running SAS')
    opts = p.parse_args()
    stack_from_groups_file(opts.groups_file, opts.spec_dir, opts.out_dir, opts.min_size, opts.dry_run,
                           {int(g) for g in opts.groups.split(',')} if opts.groups else None,
                           opts.detectors.split(','))


if __name__ == '__main__':
    main()
