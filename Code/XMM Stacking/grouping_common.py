"""Shared implementation of the XMM Galactic-disc source grouping pipeline.

Script port of stacking.ipynb (which superseded test_pipeline.py for the
GalDisc_4xmmdr14s_new_cleaned_realsrc catalog). Sources are keyed by (extinction contour,
nH contour), split into flux x spectral-shape bins, small groups are merged, and the groups
are refined by iterative sigma clipping.

Corrections relative to the notebook:
  * hardness defaults to the computed EPIC ratio EPhr2 (EP_HR4 is NaN for 96% of sources);
  * every source keeps its own hardness/quantile (the notebook copied the first source's value);
  * reassignment checks the other-quantity tolerances in every clipping pass (the notebook
    tested the wrong variable in the nH/flux/extinction passes);
  * positional clipping sigma-clips each group's member positions and picks the nearest group
    by distance (the notebook clipped a running list of group means and applied those indices
    to sources);
  * every clipping loop stops when the grouping stops changing or revisits an earlier state
    (the notebook ran a fixed 1000 passes or stopped on equal consecutive outlier counts);
  * sources brighter than 1e-11 erg/cm^2/s get their own flux bin instead of being dropped;
  * nH contour vertices are converted with the map's real WCS (--approx-nh-wcs restores the
    notebook's linear approximation).

Entry points:
    group_sources_hardness.py  -- grouping on the hardness ratio
    group_sources_quantile.py  -- same pipeline, grouping on an energy quantile
    stack_groups.py            -- stacks the grouped spectra with stacking_scripts/combine.py
"""

import csv
import json
import math
import os
import warnings

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.wcs import WCS
from matplotlib.path import Path

# ---------------------------------------------------------------------------
# Default inputs
# ---------------------------------------------------------------------------
CATALOG = '/data/tong/xmm/DR14s/GalDisc_4xmmdr14s_new_cleaned_realsrc.fits'
NH_MAP = '/data/benjamin/column_density_inner20deg_filled.fits'      # Hi-GAL N(H2), |l|<10, |b|<1
EXT_MAP = '/data/benjamin/VVVextmap_mef_ciao_ready.fits'            # VVV A_V, b in [-10.4, 6.6]
NH_CONTOURS = '/data/benjamin/test_contour_inner20deg_convolved.fits'
EXT_CONTOURS_2X20 = '/data/benjamin/VVVextmap_mef_contour_perfect.fits'
EXT_CONTOURS_22X20 = '/data/benjamin/VVVextmap_mef_contour_perfect_22x20.fits'

# The CIAO contour files store vertices in image pixels and carry no WCS, so the
# pixel -> (l, b) transform has to be supplied.
# nH: the notebook's hard-coded linear approximation, only used with --approx-nh-wcs.
# The map itself is GLON-TAN with CDELT 0.00319444, so by default contour pixels
# are converted with the map's real WCS.
NH_CONTOUR_XFORM = dict(crpix1=3164.0, crpix2=319.0, cdelt1=-0.0032, cdelt2=0.0032)
# VVV 2x20: the notebook's fallback values (0.01 deg CAR grid, 2040 x 204 px).
EXT_CONTOUR_XFORM_2X20 = dict(crpix1=1020.5, crpix2=102.5, cdelt1=-0.010, cdelt2=0.010)
# VVV 22x20: not recorded anywhere; recovered by fitting the contour vertices to
# the A_V map (median |A_V - contour level| = 0.62 mag at this solution, versus
# 0.99 mag for the 2x20 file with its known transform).
EXT_CONTOUR_XFORM_22X20 = dict(crpix1=1122.5, crpix2=1020.5, cdelt1=-0.010, cdelt2=0.010)

# ---------------------------------------------------------------------------
# Grouping parameters (all values from the notebook)
# ---------------------------------------------------------------------------
BOX_L = 10.0                 # |l| < 10 and |b| < 1: the 2x20 deg nH-map footprint
BOX_B = 1.0
NNDIST_MIN = 6.0             # arcsec; closer sources are treated as duplicates
FLUX_LABELS = ["1e-16", "1e-14", "1e-13", "1e-11", "inf"]   # upper edges; "inf" = brighter than 1e-11
HARDNESS_LABELS = ["-1", "-0.5", "0", "0.5", "1"]
HARDNESS_TOL = 0.5
SMALL_GROUP_MAX = 5          # groups with <= this many sources are re-merged
MERGE_LB_TOL = 1.0           # deg, small-group merge
REASSIGN_LB_TOL = 1.5        # deg, per-quantity sigma-clip reassignment
NH_DEX_TOL = 0.5
EXT_TOL = 7.0
FLUX_DEX_TOL = 1.7
CLIP_SIGMA = 2
CLIP_MAXITERS = 5
DIST_CLIP_MAX_ITER = 1000
MULTI_CLIP_PASSES = 2

NO_NH_COVERAGE = -2          # nh-contour sentinel for sources outside the nH map (extended mode)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
def _nansum(*arrays):
    return np.nansum(np.stack(arrays), axis=0)


def _hr(h, s):
    return (h - s) / (h + s)


def load_catalog(path=CATALOG):
    """Read the columns the pipeline uses. Row index == allspec/{index}_spec directory."""
    with fits.open(path) as hdul:
        d = hdul[1].data
        rate = {(det, band): np.array(d[f'{det}_{band}_RATE']) for det in ('PN', 'M1', 'M2') for band in (4, 5)}
        exp = {(det, band): np.array(d[f'{det}_{band}_EXP']) for det in ('PN', 'M1', 'M2') for band in (4, 5)}
        l_raw = np.array(d['LII'])
        cat = {
            'src_id': np.array(d['SRCID']),
            'l_raw': l_raw,
            'l': np.where(l_raw < 180, l_raw, l_raw - 360),
            'b': np.array(d['BII']),
            'ra': np.array(d['RA']),
            'dec': np.array(d['DEC']),
            'stack_flag': np.array(d['STACK_FLAG']),
            'extent': np.array(d['EXTENT']),
            'ndet': np.array(d['NDETECT_DETCAT']),
            'ep_exp4to12': _nansum(exp['PN', 5], exp['M1', 5], exp['M2', 5]),
            'fx2to12': np.array(d['EP_4_FLUX']) + np.array(d['EP_5_FLUX']),
            'EP_HR4': np.array(d['EP_HR4']),
            'ep_cts2to12': _nansum(*(rate[k] * exp[k] for k in rate)),
        }
    # Notebook's own EPIC HR: (4.5-12 - 2-4.5)/(sum) from summed count rates.
    # Not used by default; available as --hardness-column EPhr2.
    with np.errstate(invalid='ignore', divide='ignore'):
        cat['EPhr2'] = _hr(_nansum(rate['PN', 5], rate['M1', 5], rate['M2', 5]),
                           _nansum(rate['PN', 4], rate['M1', 4], rate['M2', 4]))
    cat['row_of'] = {int(s): i for i, s in enumerate(cat['src_id'])}
    return cat


def _dist(ra1, dec1, ra2, dec2):
    """Approximate (flat-sky) separation in degrees, as in the notebook."""
    return np.sqrt((ra2 - ra1) ** 2 + (dec2 - dec1) ** 2)


def clean_source_indices(cat, nndist_min=NNDIST_MIN):
    """Flag/extent filter, duplicate removal and 6'' nearest-neighbour cleaning (notebook cell 7)."""
    flag, ext, ndet, ids = cat['stack_flag'], cat['extent'], cat['ndet'], cat['src_id']
    ra, dec, expo = cat['ra'], cat['dec'], cat['ep_exp4to12']

    flfilt = np.where((flag == 0) & (ext == 0))[0]
    duplist = np.zeros(np.size(ndet))
    for i in flfilt:
        if ndet[i] > 1 and duplist[i] == 0:
            ilist = np.where(ids == ids[i])[0]
            jlist = np.where((flag[ilist] == 0) & (ext[ilist] == 0))[0]
            ibest = np.argmax(expo[ilist[jlist]])
            duplist[ilist] = 1
            duplist[ilist[jlist[ibest]]] = 10

    glist = np.where((flag == 0) & (ext == 0) & ((ndet == 1) | (duplist == 10)))[0]
    gra, gdec = ra[glist], dec[glist]
    nndist = np.zeros(glist.size)
    for i in range(glist.size):
        nndist[i] = 3600 * np.partition(_dist(gra, gdec, gra[i], gdec[i]), 1)[1]

    duplist2 = np.zeros(glist.size)
    for i in range(glist.size):
        if nndist[i] < nndist_min:
            ilist2 = np.where(3600 * _dist(gra, gdec, gra[i], gdec[i]) < nndist_min)[0]
            ibest2 = np.argmax(expo[glist][ilist2])
            duplist2[ilist2] = 1
            duplist2[ilist2[ibest2]] = 10

    return glist[np.where((nndist > nndist_min) | (duplist2 == 10))]


def box_indices(cat, box_l=BOX_L, box_b=BOX_B):
    l, b = cat['l'], cat['b']
    return np.where((b > -box_b) & (b < box_b) & (l > -box_l) & (l < box_l))[0]


# ---------------------------------------------------------------------------
# Maps and contours
# ---------------------------------------------------------------------------
class SkyMap:
    """A 2-D FITS map sampled at the nearest pixel (as the notebook did), with bounds checks."""

    def __init__(self, path):
        with fits.open(path) as hdul:
            hdu = hdul[0] if hdul[0].data is not None else hdul[1]
            self.data = hdu.data
            self.wcs = WCS(hdu.header)
        self.path = path

    def pixels(self, l, b):
        x, y = self.wcs.wcs_world2pix(np.atleast_1d(l), np.atleast_1d(b), 0)
        # Python round() (half-to-even) exactly as the notebook's int(round(float(x)))
        xi = np.array([int(round(float(v))) for v in x])
        yi = np.array([int(round(float(v))) for v in y])
        ny, nx = self.data.shape
        inside = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        return xi, yi, inside

    def sample(self, l, b):
        """Values at (l, b); NaN outside the map instead of numpy's silent negative-index wrap."""
        xi, yi, inside = self.pixels(l, b)
        vals = np.full(xi.size, np.nan, dtype=self.data.dtype)
        vals[inside] = self.data[yi[inside], xi[inside]]
        return vals, inside


def load_contours(path, crpix1, crpix2, cdelt1, cdelt2, lon_style, close=False, wcs=None):
    """Read a CIAO dmcontour REGION table into polygons in (l, b).

    lon_style: 'nh'  -> linear transform, no longitude wrap, polygons closed (notebook cell 14)
               'ext' -> linear transform, lon wrapped with (l+180)%360-180 (notebook cell 15)
    NOTE (preserved): the X/Y columns are NaN-padded fixed-length arrays; the padding is kept
    in the vertex list exactly as the notebook did, so contour ordering (which sorts on vertex
    count) and matplotlib's point-in-polygon behaviour are unchanged.
    """
    contours = []
    with fits.open(path) as hdul:
        for i, row in enumerate(hdul['REGION'].data):
            xs, ys = np.array(row['X']), np.array(row['Y'])
            if wcs is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    lons, lats = wcs.wcs_pix2world(xs, ys, 1)   # CIAO image pixels are 1-based
                lons = np.where(lons > 180, lons - 360, lons)
            else:
                lons = cdelt1 * (xs - crpix1)
                lats = cdelt2 * (ys - crpix2)
                if lon_style == 'ext':
                    lons = (lons + 180) % 360 - 180
            vertices = np.column_stack((lons, lats))
            if close and len(vertices) > 0 and not np.array_equal(vertices[0], vertices[-1]):
                vertices = np.vstack([vertices, vertices[0]])
            contours.append({'index': i, 'level': float(row['CONTOUR_LEVEL']), 'vertices': vertices})
    return contours


def assign_to_contours(lb, contours):
    """Contour index of the highest-level contour containing each point (-1 if none).

    Same order as the notebook: highest level first, ties by vertex count, then file order.
    """
    lb = np.asarray(lb, dtype=float)
    assigned = np.full(len(lb), -1, dtype=int)
    for contour in sorted(contours, key=lambda c: (-c['level'], len(c['vertices']))):
        todo = np.where(assigned == -1)[0]
        if todo.size == 0:
            break
        inside = Path(contour['vertices']).contains_points(lb[todo])
        assigned[todo[inside]] = contour['index']
    return assigned


def nh_source_coords(cat, idx):
    l = cat['l_raw'][idx]
    return np.column_stack((np.where(l > 180, l - 360, l), cat['b'][idx]))


def ext_source_coords(cat, idx):
    return np.column_stack(((cat['l_raw'][idx] + 180) % 360 - 180, cat['b'][idx]))


# ---------------------------------------------------------------------------
# Grouping steps
# ---------------------------------------------------------------------------
def contour_groups(cat, idx, ext_contour, nh_contour, nh_values, ext_values, status):
    """(extinction contour, nH contour) groups (notebook cell 18)."""
    result = {}
    for k, i in enumerate(idx):
        sid = int(cat['src_id'][i])
        if ext_contour[k] == -1:
            status[sid] = 'no_ext_contour'
            continue
        if nh_contour[k] == -1:
            status[sid] = 'no_nh_contour'
            continue
        key = (int(ext_contour[k]), int(nh_contour[k]))
        result.setdefault(key, []).append({
            'src_id': sid,
            'extinction_index': ext_values[k],
            'nh_index': np.float64(nh_values[k]),
        })
    return result


def hardness_binner(labels=HARDNESS_LABELS):
    """Notebook cell 22 binning: exact 1 and 0 get their own labels, then first label >= value."""
    def binner(h):
        if h == 1:
            return '1+'
        if h == 0:
            return '0_exact'
        if math.isnan(h):
            return 'nan'
        for label in labels:
            if h <= float(label):
                return label
        return None
    return binner


def edge_binner(edges, fmt='{:.4f}'):
    """First upper edge >= value; NaN -> 'nan'; above the last edge -> None (dropped)."""
    labels = [fmt.format(e) for e in edges]

    def binner(q):
        if q is None or math.isnan(q):
            return 'nan'
        for label, edge in zip(labels, edges):
            if q <= edge:
                return label
        return None
    return binner


def shape_bins(result, shape_values, binner, status, shape_key):
    """Spectral-shape bins inside each contour group (notebook cell 22, generalised)."""
    shape_dict = {}
    for key, sources in result.items():
        shape_dict[key] = {}
        for source in sources:
            value = shape_values[source['src_id']]
            label = binner(value)
            if label is None:
                status[source['src_id']] = f'{shape_key}_out_of_range'
                continue
            shape_dict[key].setdefault(label, []).append({'src_id': source['src_id'], 'value': value})
    return shape_dict


def flux_bins(result, flux_of, status, labels=FLUX_LABELS):
    """Flux bins inside each contour group (notebook cell 23, plus an open top bin above 1e-11)."""
    flux_dict = {}
    for key, sources in result.items():
        flux_dict[key] = {}
        for source in sources:
            flux = flux_of[source['src_id']]
            if math.isnan(flux):
                flux_dict[key].setdefault('nan', []).append({'src_id': source['src_id'], 'flux': flux})
                continue
            for label in labels:
                if flux <= float(label):
                    flux_dict[key].setdefault(label, []).append({
                        'src_id': source['src_id'],
                        'nh': source['nh_index'],
                        'extinction': source['extinction_index'],
                        'flux': flux,
                    })
                    break
    return flux_dict


def combine_flux_and_shape(flux_dict, shape_dict, shape_key, status):
    """Cross flux bins with shape bins (notebook cell 24). Each source keeps its own shape value
    (the notebook gave every source the value of the first source in its bin); NaN-shape sources
    are dropped."""
    combined = {}
    for key, inner in flux_dict.items():
        combined[key] = {}
        label_of = {e['src_id']: (label, e['value']) for label, entries in shape_dict[key].items() for e in entries}
        for flux_label, sources in inner.items():
            for source in sources:
                sid = source['src_id']
                if sid not in label_of:    # already dropped as out of range
                    continue
                label, shape = label_of[sid]
                if math.isnan(shape):
                    status[sid] = f'nan_{shape_key}'
                    continue
                if 'nh' not in source:
                    # NaN flux: the notebook would raise KeyError here; drop the source instead.
                    status[sid] = 'nan_flux'
                    continue
                combined[key].setdefault((flux_label, label), []).append({
                    'src_id': sid,
                    'nh': source['nh'],
                    'extinction': source['extinction'],
                    'flux': source['flux'],
                    shape_key: shape,
                })
    return combined


def merge_small_groups(data, lb, shape_key, shape_tol):
    """Merge members of groups with <= SMALL_GROUP_MAX sources with nearby, similar ones (cell 25).

    Each candidate is compared with the seed source only (not the growing group), as in the notebook.
    """
    small = []
    for outer_key, inner in data.items():
        for inner_key, sources in inner.items():
            if isinstance(sources, list) and len(sources) <= SMALL_GROUP_MAX:
                for source in sources:
                    small.append({'outer_key': outer_key, 'inner_key': inner_key, 'source': source})

    merged = []
    used = set()
    n_joined = 0
    with np.errstate(invalid='ignore', divide='ignore'):
        for i, entry in enumerate(small):
            s1 = entry['source']
            if s1['src_id'] in used:
                continue
            group = [entry]
            used.add(s1['src_id'])
            l1, b1 = lb[s1['src_id']]
            for j, other in enumerate(small):
                s2 = other['source']
                if j == i or s2['src_id'] in used:
                    continue
                l2, b2 = lb[s2['src_id']]
                if abs(l1 - l2) > MERGE_LB_TOL or abs(b1 - b2) > MERGE_LB_TOL:
                    continue
                if abs(np.log10(s1['nh']) - np.log10(s2['nh'])) > NH_DEX_TOL:
                    continue
                if abs(s1['extinction'] - s2['extinction']) > EXT_TOL:
                    continue
                if abs(np.log10(s1['flux']) - np.log10(s2['flux'])) > FLUX_DEX_TOL:
                    continue
                if abs(s1[shape_key] - s2[shape_key]) > shape_tol:
                    continue
                group.append(other)
                used.add(s2['src_id'])
                n_joined += 1
            merged.append(group)

    merged_dict = {}
    for group in merged:
        outer_keys = tuple(sorted(set(e['outer_key'] for e in group)))
        inner_keys = tuple(sorted(set(e['inner_key'] for e in group)))
        merged_dict.setdefault(outer_keys, {}).setdefault(inner_keys, []).extend(e['source'] for e in group)
    stats = {'small_entries': sum(len(g) for g in merged), 'used': len(used), 'joined': n_joined}
    return merged_dict, stats


def replace_small_groups(combined, merged_small):
    """Drop the small groups and add the re-merged ones (cell 26)."""
    kept = {}
    for outer_key, inner in combined.items():
        new_inner = {k: v for k, v in inner.items() if not (isinstance(v, list) and len(v) <= SMALL_GROUP_MAX)}
        if new_inner:
            kept[outer_key] = new_inner
    out = kept.copy()
    for outer_key, inner in merged_small.items():
        out.setdefault(outer_key, {})
        for inner_key, sources in inner.items():
            out[outer_key][inner_key] = sources
    return out


# --- sigma clipping --------------------------------------------------------
def _mean(values):
    """np.mean, falling back to nanmean when some values are NaN (only happens outside map coverage)."""
    arr = np.asarray(values)
    if arr.dtype.kind == 'f' and np.isnan(arr).any():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanmean(arr)
    return np.mean(values)


def _within(diff, tol):
    """diff < tol, treating a NaN difference (quantity unavailable for a source) as compatible."""
    return diff < tol or math.isnan(diff)


def _record(source, shape_key):
    return {'src_id': source['src_id'], 'nh': source['nh'], 'extinction': source['extinction'],
            'flux': source['flux'], shape_key: source[shape_key]}


def _group_meta(sources, lb, shape_key):
    return {
        'mean_l': np.mean([lb[s['src_id']][0] for s in sources]),
        'mean_b': np.mean([lb[s['src_id']][1] for s in sources]),
        f'mean_{shape_key}': _mean([s[shape_key] for s in sources]),
        'mean_flux': _mean([s['flux'] for s in sources]),
        'mean_nH': _mean([s['nh'] for s in sources]),
        'mean_extinction': _mean([s['extinction'] for s in sources]),
    }


def _outlier_indices(values):
    """Indices of sigma-clipped values; NaN entries (quantity unavailable) are never outliers."""
    values = np.asarray(values, dtype=np.float64)
    if np.isfinite(values).sum() < 3:
        return set()
    clipped = sigma_clip(values, sigma=CLIP_SIGMA, maxiters=CLIP_MAXITERS)
    return set(np.where(np.asarray(clipped.mask) & np.isfinite(values))[0].tolist())


def _quantity_diff(quantity, source, meta, shape_key):
    """Distance between a source and a group mean in the units the tolerances are defined in."""
    if quantity == 'nh':
        return abs(np.log10(np.float64(source['nh'])) - np.log10(np.float64(meta['mean_nH'])))
    if quantity == 'flux':
        return abs(np.log10(np.float64(source['flux'])) - np.log10(np.float64(meta['mean_flux'])))
    if quantity == 'extinction':
        return abs(source['extinction'] - meta['mean_extinction'])
    return abs(source[shape_key] - meta[f'mean_{shape_key}'])


def _compatible(source, meta, shape_key, shape_tol, skip=None):
    """All tolerances (nH 0.5 dex, extinction 7, flux 1.7 dex, shape) except `skip`."""
    tols = {'nh': NH_DEX_TOL, 'extinction': EXT_TOL, 'flux': FLUX_DEX_TOL, shape_key: shape_tol}
    return all(_within(_quantity_diff(q, source, meta, shape_key), tol) for q, tol in tols.items() if q != skip)


def _as_groups(data, shape_key):
    """{outer: {inner: [sources]}} -> {outer: {inner: {'sources': [...]}}}."""
    return {k: {ik: {'sources': [_record(s, shape_key) for s in (g['sources'] if isinstance(g, dict) else g)]}
                for ik, g in inner.items()} for k, inner in data.items()}


def _signature(groups):
    return frozenset((k, ik, frozenset(s['src_id'] for s in g['sources']))
                     for k, inner in groups.items() for ik, g in inner.items() if g['sources'])


def _clip_and_reassign(groups, lb, shape_key, shape_tol, outliers_of, candidate_dist, lb_limit, skip):
    """One pass: clip outliers (and single-source groups) out of every group, then move each
    clipped source to the nearest remaining group if it is compatible, else back to its own group."""
    clipped, remain, n_out = [], {}, 0
    for key, inner in groups.items():
        for ik, g in inner.items():
            members = g['sources']
            out = {0} if len(members) == 1 else outliers_of(members)
            if len(members) > 1:
                n_out += len(out)
            keep = [s for i, s in enumerate(members) if i not in out]
            clipped += [(key, ik, s) for i, s in enumerate(members) if i in out]
            if keep:
                remain.setdefault(key, {})[ik] = {'sources': keep, 'meta': _group_meta(keep, lb, shape_key)}

    new = {k: {ik: {'sources': list(g['sources'])} for ik, g in inner.items()} for k, inner in remain.items()}
    n_no_candidate = 0
    for key, ik, source in clipped:
        l, b = lb[source['src_id']]
        best, best_d = None, math.inf
        for k2, inner in remain.items():
            for ik2, g in inner.items():
                m = g['meta']
                if lb_limit is not None and (abs(l - m['mean_l']) >= lb_limit or abs(b - m['mean_b']) >= lb_limit):
                    continue
                d = candidate_dist(source, m)
                if d < best_d:
                    best, best_d = (k2, ik2), d
        if best is None:
            n_no_candidate += 1
        if best is not None and _compatible(source, remain[best[0]][best[1]]['meta'], shape_key, shape_tol, skip):
            new[best[0]][best[1]]['sources'].append(source)
        else:
            new.setdefault(key, {}).setdefault(ik, {'sources': []})['sources'].append(source)
    return new, n_out, n_no_candidate


def _iterate(groups, pass_fn, max_iter):
    """Repeat pass_fn until no outliers remain, the grouping stops changing, or it revisits an
    earlier state (a cycle), capped at max_iter passes."""
    seen = {_signature(groups)}
    history = []
    with np.errstate(invalid='ignore', divide='ignore'):
        for _ in range(max_iter):
            groups, n_out, n_no_candidate = pass_fn(groups)
            history.append((n_out, n_no_candidate))
            sig = _signature(groups)
            if sig in seen:
                break
            seen.add(sig)
    return groups, history


def distance_sigma_clip(final_test, lb, shape_key, shape_tol, max_iter=DIST_CLIP_MAX_ITER, log=None):
    """Positional sigma clipping (notebook cell 27, corrected).

    In each group the member positions l and b are sigma clipped (2 sigma); outliers and the
    members of single-source groups move to the nearest group (by sqrt(dl^2 + db^2) to the group
    mean) if nH, extinction, flux and shape are within tolerance, otherwise they stay put.
    """
    def outliers_of(members):
        return (_outlier_indices([lb[s['src_id']][0] for s in members])
                | _outlier_indices([lb[s['src_id']][1] for s in members]))

    def candidate_dist(source, m):
        l, b = lb[source['src_id']]
        return math.hypot(l - m['mean_l'], b - m['mean_b'])

    def one_pass(groups):
        return _clip_and_reassign(groups, lb, shape_key, shape_tol, outliers_of, candidate_dist, None, None)

    groups, history = _iterate(_as_groups(final_test, shape_key), one_pass, max_iter)
    if log:
        log(f'  distance clip: {len(history)} passes, outliers per pass {[h[0] for h in history][-5:]}')
    return groups, history


def quantity_sigma_clip(start, lb, shape_key, shape_tol, passes=MULTI_CLIP_PASSES, log=None):
    """Per-quantity sigma clipping: nH, flux, extinction, shape; repeated `passes` times (cell 28, corrected).

    For each quantity, outliers in that quantity move to the group within 1.5 deg whose mean is
    closest in that quantity, provided the other three quantities are within tolerance.
    """
    groups = _as_groups(start, shape_key)
    stats = []
    for _ in range(passes):
        for quantity in ['nh', 'flux', 'extinction', shape_key]:
            def outliers_of(members, q=quantity):
                return _outlier_indices([s[q] for s in members])

            def candidate_dist(source, m, q=quantity):
                d = _quantity_diff(q, source, m, shape_key)
                return math.inf if math.isnan(d) else d

            def one_pass(g, oo=outliers_of, cd=candidate_dist, q=quantity):
                return _clip_and_reassign(g, lb, shape_key, shape_tol, oo, cd, REASSIGN_LB_TOL, q)

            groups, history = _iterate(groups, one_pass, 1000)
            stats.append({'quantity': quantity, 'iterations': len(history),
                          'outliers': [h[0] for h in history],
                          'no_candidate_within_1.5deg': sum(h[1] for h in history)})
            if log:
                log(f'  {quantity} clip: iterations={len(history)} outliers={stats[-1]["outliers"][-5:]} '
                    f'no_candidate_within_{REASSIGN_LB_TOL}deg={stats[-1]["no_candidate_within_1.5deg"]}')
    for inner in groups.values():
        for g in inner.values():
            g['meta'] = [_group_meta(g['sources'], lb, shape_key)] if g['sources'] else []
    return groups, stats


# ---------------------------------------------------------------------------
# Accounting / export
# ---------------------------------------------------------------------------
def count_sources(groups, nested=True):
    n = 0
    for inner in groups.values():
        for g in inner.values():
            n += len(g['sources'] if nested else g)
    return n


def duplicate_count(groups):
    seen, dups = set(), set()
    for inner in groups.values():
        for g in inner.values():
            for s in g['sources']:
                (dups if s['src_id'] in seen else seen).add(s['src_id'])
    return len(dups)


def group_size_summary(groups):
    sizes = [len(g['sources']) for inner in groups.values() for g in inner.values()]
    multi = [n for n in sizes if n > 1]
    return {
        'n_groups': len(sizes),
        'n_single_source_groups': sum(1 for n in sizes if n == 1),
        'n_multi_source_groups': len(multi),
        'n_sources_in_multi_groups': int(sum(multi)),
        'median_multi_size': float(np.median(multi)) if multi else None,
        'mean_multi_size': float(np.mean(multi)) if multi else None,
        'max_size': int(max(sizes)) if sizes else 0,
        'min_multi_size': int(min(multi)) if multi else None,
    }


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def export_groups(out_dir, final, cat, shape_key, status, config, stage_stats, min_list_size=2):
    """Write groups.json, per-group index lists for combine.py, source_status.csv and a FITS table."""
    os.makedirs(out_dir, exist_ok=True)
    lists_dir = os.path.join(out_dir, 'group_lists')
    os.makedirs(lists_dir, exist_ok=True)

    groups = []
    group_of = {}
    gid = 0
    for outer_key, inner in final.items():
        for inner_key, g in inner.items():
            if not g['sources']:
                continue
            members = []
            for s in g['sources']:
                row = cat['row_of'][s['src_id']]
                group_of[s['src_id']] = gid
                members.append({
                    'src_id': s['src_id'], 'catalog_index': row,
                    'l': cat['l'][row], 'b': cat['b'][row],
                    'nh': s['nh'], 'extinction': s['extinction'], 'flux': s['flux'],
                    shape_key: s[shape_key], f'own_{shape_key}': config['own_shape'].get(s['src_id']),
                })
            rows = [m['catalog_index'] for m in members]
            groups.append({
                'group_id': gid, 'outer_key': str(outer_key), 'inner_key': str(inner_key),
                'n_sources': len(members),
                'summary': {
                    'mean_l': float(np.mean(cat['l'][rows])), 'mean_b': float(np.mean(cat['b'][rows])),
                    'mean_nh': _mean([m['nh'] for m in members]),
                    'mean_extinction': _mean([m['extinction'] for m in members]),
                    'mean_flux': _mean([m['flux'] for m in members]),
                    f'mean_{shape_key}': _mean([m[shape_key] for m in members]),
                    'total_flux_2to12': float(np.nansum(cat['fx2to12'][rows])),
                    'total_epic_counts_2to12': float(np.nansum(cat['ep_cts2to12'][rows])),
                },
                'sources': members,
            })
            if len(members) >= min_list_size:
                with open(os.path.join(lists_dir, f'group_{gid:04d}.txt'), 'w') as f:
                    f.write('\n'.join(str(r) for r in rows) + '\n')
            gid += 1

    cfg = {k: v for k, v in config.items() if k != 'own_shape'}
    with open(os.path.join(out_dir, 'groups.json'), 'w') as f:
        json.dump(_jsonable({'config': cfg, 'stages': stage_stats, 'summary': group_size_summary(final),
                             'groups': groups}), f, indent=1)

    with open(os.path.join(out_dir, 'source_status.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['catalog_index', 'src_id', 'l', 'b', 'status', 'group_id'])
        for row, sid in enumerate(cat['src_id']):
            sid = int(sid)
            st = 'grouped' if sid in group_of else status.get(sid, 'unknown')
            w.writerow([row, sid, f"{cat['l'][row]:.6f}", f"{cat['b'][row]:.6f}", st, group_of.get(sid, -1)])

    gcol = np.array([group_of.get(int(s), -1) for s in cat['src_id']], dtype=np.int32)
    with fits.open(config['catalog']) as hdul:
        cols = hdul[1].columns + fits.ColDefs([fits.Column(name='GROUP_ID', format='J', array=gcol)])
        fits.BinTableHDU.from_columns(cols, header=hdul[1].header).writeto(
            os.path.join(out_dir, 'catalog_with_groups.fits'), overwrite=True)
    return groups


def write_group_regions(out_dir, groups):
    """One DS9 region file per group (galactic circles, 20''), as test_pipeline.py did."""
    reg_dir = os.path.join(out_dir, 'regions')
    os.makedirs(reg_dir, exist_ok=True)
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange',
              'pink', 'lime', 'purple', 'brown', 'gray', 'olive', 'navy', 'teal']
    for g in groups:
        with open(os.path.join(reg_dir, f"group_{g['group_id']:04d}.reg"), 'w') as f:
            f.write('# Region file format: DS9 version 4.1\n')
            f.write(f"global color={colors[g['group_id'] % len(colors)]} width=1\ngalactic\n")
            for s in g['sources']:
                f.write(f'circle({s["l"]:.6f},{s["b"]:.6f},20") # text={{{g["group_id"]}}}\n')


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def add_common_args(parser):
    parser.add_argument('--catalog', default=CATALOG)
    parser.add_argument('--out', help='output directory')
    parser.add_argument('--coverage', choices=['footprint', 'extended'], default='footprint',
                        help="footprint (default, as the notebook): only clean sources inside the 2x20 deg "
                             "nH map (|l|<10, |b|<1). extended: all clean sources; extinction contours from the "
                             "22x20 deg VVV contour file; sources outside the nH map are grouped with nH treated "
                             "as unavailable (nh-contour key %d, nH tolerances skipped)." % NO_NH_COVERAGE)
    parser.add_argument('--approx-nh-wcs', action='store_true',
                        help="convert nH contour pixels with the notebook's linear 0.0032 deg/px approximation "
                             "instead of the map's real GLON-TAN WCS (off by up to ~0.1 deg at |l|~10)")
    parser.add_argument('--regions', action='store_true', help='also write one DS9 region file per group')
    parser.add_argument('--min-list-size', type=int, default=2,
                        help='write group_lists/*.txt (combine.py input) only for groups with at least this many sources')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--stack', action='store_true', help='run stack_groups.py on the result')
    parser.add_argument('--spec-dir', default=None, help='per-source spectra base dir (stack_groups default)')
    parser.add_argument('--stack-out', default=None, help='stacked spectra output dir')
    parser.add_argument('--dry-run', action='store_true', help='with --stack: match files but do not run SAS')


def run_grouping(opts, shape_key, shape_values_for, make_binner, shape_tol_for, extra_config=None):
    """Full grouping run.

    shape_values_for(cat) -> {src_id: value}   spectral-shape value per source (NaN if unavailable)
    make_binner(values)   -> binner            values: shape values of the contour-assigned sources
    shape_tol_for(values) -> float             tolerance used in the merge / clip steps
    """
    log = print
    cat = load_catalog(opts.catalog)
    status = {}
    stages = {}

    clean = clean_source_indices(cat)
    clean_set = set(clean.tolist())
    for i, sid in enumerate(cat['src_id']):
        if i not in clean_set:
            status[int(sid)] = 'not_clean'
    log(f'catalog rows: {len(cat["src_id"])}, clean: {clean.size}')

    if opts.coverage == 'footprint':
        idx = np.intersect1d(clean, box_indices(cat))
        for i in np.setdiff1d(clean, idx):
            status[int(cat['src_id'][i])] = 'outside_2x20_nh_map'
        ext_file, ext_xform = EXT_CONTOURS_2X20, EXT_CONTOUR_XFORM_2X20
    else:
        idx = clean
        ext_file, ext_xform = EXT_CONTOURS_22X20, EXT_CONTOUR_XFORM_22X20
    log(f'sources considered ({opts.coverage}): {idx.size}; outside: {clean.size - idx.size}')
    stages['n_clean'] = int(clean.size)
    stages['n_considered'] = int(idx.size)

    nh_map, ext_map = SkyMap(NH_MAP), SkyMap(EXT_MAP)
    nh_vals, in_nh = nh_map.sample(cat['l'][idx], cat['b'][idx])
    ext_vals, in_ext = ext_map.sample(cat['l'][idx], cat['b'][idx])
    stages['n_outside_nh_map'] = int((~in_nh).sum())
    stages['n_outside_ext_map'] = int((~in_ext).sum())
    stages['n_nan_extinction_value'] = int(np.isnan(ext_vals[in_ext]).sum())
    log(f'outside nH map: {stages["n_outside_nh_map"]}, outside extinction map: {stages["n_outside_ext_map"]}, '
        f'NaN extinction pixel: {stages["n_nan_extinction_value"]}')

    log('loading contours ...')
    nh_contours = load_contours(NH_CONTOURS, lon_style='nh', close=True,
                                wcs=None if opts.approx_nh_wcs else nh_map.wcs, **NH_CONTOUR_XFORM)
    nh_contour = assign_to_contours(nh_source_coords(cat, idx), nh_contours)
    del nh_contours
    ext_contours = load_contours(ext_file, lon_style='ext', **ext_xform)
    ext_contour = assign_to_contours(ext_source_coords(cat, idx), ext_contours)
    del ext_contours
    if opts.coverage == 'extended':
        nh_contour[~in_nh] = NO_NH_COVERAGE
    stages['n_no_ext_contour'] = int((ext_contour == -1).sum())
    stages['n_no_nh_contour'] = int((nh_contour == -1).sum())
    log(f'unassigned: extinction contour {stages["n_no_ext_contour"]}, nH contour {stages["n_no_nh_contour"]}')

    result = contour_groups(cat, idx, ext_contour, nh_contour, nh_vals, ext_vals, status)
    stages['n_contour_groups'] = len(result)
    stages['n_in_contour_groups'] = sum(len(v) for v in result.values())

    shape_all = shape_values_for(cat)
    in_result = np.array([shape_all[s['src_id']] for v in result.values() for s in v], dtype=float)
    binner, bin_info = make_binner(in_result)
    shape_tol = shape_tol_for(in_result)
    stages['shape_binning'] = bin_info
    stages['shape_tol'] = shape_tol
    log(f'{shape_key}: {np.isfinite(in_result).sum()} of {in_result.size} sources have a value; '
        f'bins {bin_info}; tolerance {shape_tol}')

    flux_of = {int(s): f for s, f in zip(cat['src_id'], cat['fx2to12'])}
    shape_dict = shape_bins(result, shape_all, binner, status, shape_key)
    flux_dict = flux_bins(result, flux_of, status)
    combined = combine_flux_and_shape(flux_dict, shape_dict, shape_key, status)
    stages['n_shape_bins'] = sum(len(v) for v in shape_dict.values())
    stages['n_flux_bins'] = sum(len(v) for v in flux_dict.values())
    stages['n_flux_shape_bins'] = sum(len(v) for v in combined.values())
    n_grouped = count_sources(combined, nested=False)
    stages['n_after_flux_shape'] = n_grouped
    log(f'contour groups {len(result)}, {shape_key} bins {stages["n_shape_bins"]}, flux bins '
        f'{stages["n_flux_bins"]}, flux x {shape_key} bins {stages["n_flux_shape_bins"]}, sources {n_grouped}')

    lb = {int(s): (cat['l'][i], cat['b'][i]) for i, s in enumerate(cat['src_id'])}
    merged_small, mstats = merge_small_groups(combined, lb, shape_key, shape_tol)
    stages['merge_small'] = mstats
    final_test = replace_small_groups(combined, merged_small)
    log(f'small-group merge: {mstats}')
    assert count_sources(final_test, nested=False) == n_grouped

    log('distance sigma clipping ...')
    dist_dict, history = distance_sigma_clip(final_test, lb, shape_key, shape_tol,
                                             log=log if opts.verbose else None)
    stages['distance_clip'] = {'passes': len(history), 'first': history[:5], 'last': history[-1:]}
    log(f'  passes {len(history)}, (flagged, singles) first {history[:4]} last {history[-1]}')
    assert count_sources(dist_dict) == n_grouped and duplicate_count(dist_dict) == 0

    log('per-quantity sigma clipping ...')
    final, qstats = quantity_sigma_clip(dist_dict, lb, shape_key, shape_tol, log=log)
    stages['quantity_clip'] = qstats
    assert count_sources(final) == n_grouped and duplicate_count(final) == 0

    summary = group_size_summary(final)
    log(f'final: {summary}')

    config = {
        'catalog': opts.catalog, 'coverage': opts.coverage, 'approx_nh_wcs': opts.approx_nh_wcs,
        'nh_map': NH_MAP, 'ext_map': EXT_MAP, 'nh_contours': NH_CONTOURS, 'ext_contours': ext_file,
        'shape_key': shape_key, 'shape_tol': shape_tol, 'own_shape': shape_all,
    }
    config.update(extra_config or {})
    groups = export_groups(opts.out, final, cat, shape_key, status, config, stages, opts.min_list_size)
    if opts.regions:
        write_group_regions(opts.out, groups)
    log(f'wrote {len(groups)} groups to {opts.out}')

    if opts.stack:
        import stack_groups
        stack_groups.stack_from_groups_file(os.path.join(opts.out, 'groups.json'),
                                            spec_dir=opts.spec_dir, out_dir=opts.stack_out,
                                            min_size=opts.min_list_size, dry_run=opts.dry_run)
    return final, groups
