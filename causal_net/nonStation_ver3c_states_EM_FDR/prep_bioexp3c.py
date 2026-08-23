#!/usr/bin/env python3
"""
Preprocessing pipeline for experimental neural data from Roy/Mandar laboratory.

This script processes raw experimental neural recordings into standardized format
for connectivity analysis. The preprocessing pipeline includes:
- Raw data loading and format conversion
- Temporal binning and spike count extraction
- Data quality assessment and filtering
- Metadata extraction and session identification
- Metrics spreadsheet ingestion (metrics_curated.xlsx)
- Output formatting for downstream analysis tools

Session naming convention:
- B6J: cell line name
- 250619: recording date (YYMMDD)
- M08020: chip identifier
- 000093: run number
- Well000: well number

The script generates .spikes.npz files with standardized spike count matrices
and .bioExp.npz files containing experimental metadata and raw mean waveforms.

Usage:
    ./prep_bioexp.py --sessionName B6J_250619_M08020_000093_Well000 --inputPath /path/to/raw/data/
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
import sys, os, hashlib
import numpy as np
import pandas as pd
from pprint import pprint
from toolbox.Util_NumpyIOv2 import json_safe_metadata, write_data_npz

import argparse


BIOEXP_SCHEMA_VERSION = 3
WAVEFORM_GRID_PITCH = 17.5
WAVEFORM_GRID_ORIGIN = np.asarray([0.0, 0.0], dtype=np.float64)
WAVEFORM_MULTICHANNEL_DISTANCE_THRESHOLD = 1.0


#...!...!..................
def commandline_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verb", type=int, help="increase debug verbosity", default=1)
    parser.add_argument("--expPath", required=True, help="raw experimnetal data on CFS")

    parser.add_argument("--sessionName", default='celllinename/dateofrecording/chipID/Assaytype/runnumber/wellnumber', help='raw data session name')
    parser.add_argument("--dataPath", default='/pscratch/sd/b/balewski/2025_causalNet_tmp/', help="head dir for any further data processing")
    parser.add_argument("--shortName", default=None, help='(optional) output file name - Is it needed?')
    parser.add_argument("--inpFormat", default=1, type=int, choices=[1, 2],
                        help="input format selector: 1=metrics_curated.xlsx, 2=quality_metrics.xlsx")

    # .... activity speciffic speciffic,
    parser.add_argument('--samp_freq', default=100, type=int, help='sets binning of time axis')

    parser.add_argument('--freqRange', default=[1., 50], type=float, nargs=2, help='rebin of raw time axis')

    args = parser.parse_args()

    for arg in vars(args):
        print('myArgs:', arg, getattr(args, arg))

    return args


#...!...!....................
def buildBioMeta(args):
    pd = {}  # payload
    pd['raw_bioexp_path'] = args.expPath
    pd['session_name'] = args.sessionName
    pd['inp_format'] = args.inpFormat
    txtL = args.sessionName.split('/')
    #print('tt',txtL); aa
    pd['cell_line_name'] = txtL[0]
    pd['recording_date'] = txtL[1]
    pd['chip_ID'] = txtL[2]
    pd['run_num'] = txtL[4]
    pd['well_num'] = txtL[5]

    sel = {'freq_range': args.freqRange}
    md = {'bioexp': pd, 'data_selector': sel}
    myHN = hashlib.md5(os.urandom(32)).hexdigest()[:6]
    md['hash'] = myHN
    if args.shortName == None:
        md['short_name'] = '%s-%s' % (pd['recording_date'], md['hash'])
    else:
        md['short_name'] = args.shortName

    if args.verb > 1:
        print('\nBMD:')
        pprint(md)
    return md


def read_spike_npy(md, args):
    pmd = md['bioexp']
    inpF = os.path.join(args.expPath, args.sessionName, 'spike_times.npy')
    print('inpF:', inpF)
    assert os.path.exists(inpF)
    # Load the dictionary from the .npy file
    spike_dict = np.load(inpF, allow_pickle=True).item()

    pmd['sampling_freq'] = args.samp_freq

    # neuron ID  MEA chip
    meaIdL = np.array(sorted(spike_dict))  # here order of feature_id is settled
    maxFeat = len(meaIdL)

    if args.verb > 1:
        print('RSN: meaID list:', meaIdL)
    pmd['num_feature'] = len(meaIdL)

    spikeT = {}  # spike times
    spikeCntL = np.zeros(pmd['num_feature'], dtype=int)  # num spikes per  neuron
    maxTbin = 0

    j = 0
    for k in meaIdL:
        rec = np.array(spike_dict[k]) * args.samp_freq
        #print(rec[:100],len(rec))
        spikeT[k] = rec.astype(int)  # time-bin may repeat
        spikeCntL[j] = len(rec)
        j += 1
        if len(rec) == 0:
            continue
        mxTb = np.max(rec)
        if maxTbin < mxTb:
            maxTbin = mxTb
        #exit(0)
    pmd['num_time_bin'] = int(maxTbin) + 1
    pmd['max_time'] = pmd['num_time_bin'] / pmd['sampling_freq']
    chanFreq = spikeCntL / pmd['max_time']
    rawD = {'spikeT': spikeT, 'chanFreq': chanFreq,
            'spike_key': meaIdL, 'MEA_idx': meaIdL.copy()}
    return rawD


def load_raw_mean_templates(args):
    """Load the per-unit raw mean waveform records produced by the sorter."""
    inpF = os.path.join(args.expPath, args.sessionName, "raw_mean_templates.npy")
    print("waveform npy:", inpF)
    if not os.path.isfile(inpF):
        raise FileNotFoundError(f"missing waveform file: {inpF}")

    container = np.load(inpF, allow_pickle=True)
    if container.shape != () or container.dtype != object:
        raise ValueError(
            f"{inpF} must contain a scalar NumPy object holding a dictionary"
        )
    templates = container.item()
    if not isinstance(templates, dict):
        raise TypeError(f"expected a dictionary in {inpF}")
    return templates, inpF


def _mea_match_key(val):
    """Canonical key for matching MEA_idx across npy dict keys and spreadsheet."""
    if isinstance(val, (bytes, np.bytes_)):
        val = val.decode()
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        if np.isnan(val):
            raise ValueError("MEA_idx is NaN in metrics spreadsheet")
        f = float(val)
        return str(int(f)) if f == int(f) else str(f)
    s = str(val).strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else s
    except ValueError:
        return s


def metrics_xlsx_name(args):
    return "quality_metrics.xlsx" if args.inpFormat == 2 else "metrics_curated.xlsx"


def _read_metrics_frame(args):
    metrics_name = metrics_xlsx_name(args)
    xlsxF = os.path.join(args.expPath, args.sessionName, metrics_name)
    print("metrics xlsx:", xlsxF)
    assert os.path.exists(xlsxF), f"missing metrics spreadsheet: {xlsxF}"

    df = pd.read_excel(xlsxF, engine="openpyxl")
    cols = list(df.columns)
    cols[0] = "MEA_idx"
    df.columns = cols

    if args.inpFormat == 2:
        rename_map = {
            "location_X": "loc_x",
            "location_Y": "loc_y",
            "location_Z": "loc_z",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    else:
        assert "MEA_idx" in df.columns, "first spreadsheet column must be MEA_idx"

    return df, metrics_name


def _is_integral_key(val):
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return False
    return np.isfinite(fval) and fval == int(fval)


def _integer_if_possible(vals):
    arr = np.asarray(vals)
    try:
        farr = arr.astype(np.float64)
    except (TypeError, ValueError):
        return arr
    if np.all(np.isfinite(farr)) and np.all(farr == farr.astype(np.int64)):
        return farr.astype(np.int64)
    return arr


def classify_waveform_channel_construction(node_positions):
    """Classify multi-channel units by distance from the fixed electrode grid."""
    node_positions = np.asarray(node_positions, dtype=np.float64)
    if node_positions.ndim != 2 or node_positions.shape[1] != 2:
        raise ValueError(
            "node_positions must have shape (num_units, 2), got "
            f"{node_positions.shape}"
        )
    if not np.all(np.isfinite(node_positions)):
        raise ValueError("node_positions contains non-finite coordinates")

    nearest_grid = WAVEFORM_GRID_ORIGIN + np.rint(
        (node_positions - WAVEFORM_GRID_ORIGIN) / WAVEFORM_GRID_PITCH
    ) * WAVEFORM_GRID_PITCH
    grid_distance = np.linalg.norm(node_positions - nearest_grid, axis=1)
    is_multichannel = (
        grid_distance > WAVEFORM_MULTICHANNEL_DISTANCE_THRESHOLD
    )
    return grid_distance.astype(np.float64), is_multichannel.astype(bool)


def add_raw_mean_waveforms(bioD, bioMD, templates, source_file):
    """Add waveform records aligned with the frequency-sorted neuron axis."""
    template_by_unit = {}
    for template_key, record in templates.items():
        key = _mea_match_key(template_key)
        if key in template_by_unit:
            raise ValueError(
                f"duplicate waveform unit ID {template_key!r} in {source_file}"
            )
        if not isinstance(record, dict):
            raise TypeError(
                f"waveform record for unit {template_key!r} must be a dictionary"
            )
        template_by_unit[key] = record

    unit_ids = np.asarray(bioD["spike_key"]).ravel()
    mea_ids = np.asarray(bioD["MEA_idx"]).ravel()
    if unit_ids.size != mea_ids.size or any(
        _mea_match_key(unit_id) != _mea_match_key(mea_id)
        for unit_id, mea_id in zip(unit_ids, mea_ids)
    ):
        raise ValueError("spike_key and MEA_idx are not aligned by unit ID")
    records = []
    waveforms = []
    for unit_id in unit_ids:
        key = _mea_match_key(unit_id)
        if key not in template_by_unit:
            raise KeyError(
                f"selected unit {unit_id!r} has no waveform in {source_file}"
            )
        record = template_by_unit[key]
        if "raw_mean_template" not in record or "primary_channel" not in record:
            raise KeyError(
                f"waveform record for unit {unit_id!r} must contain "
                "raw_mean_template and primary_channel"
            )
        if "unit_id" in record and _mea_match_key(record["unit_id"]) != key:
            raise ValueError(
                f"waveform key {unit_id!r} disagrees with record unit_id "
                f"{record['unit_id']!r}"
            )

        waveform = np.asarray(record["raw_mean_template"], dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError(
                f"unit {unit_id!r} waveform must be a non-empty 1D array, "
                f"got shape {waveform.shape}"
            )
        records.append(record)
        waveforms.append(waveform)

    num_samples = np.asarray([waveform.size for waveform in waveforms], dtype=np.int32)
    max_samples = int(np.max(num_samples))
    waveform_2d = np.full(
        (len(waveforms), max_samples), np.nan, dtype=np.float32
    )
    for index, waveform in enumerate(waveforms):
        waveform_2d[index, : waveform.size] = waveform

    channel_ids = _integer_if_possible(
        [record["primary_channel"] for record in records]
    )
    ms_before = np.asarray(
        [record.get("ms_before", 0.0) for record in records], dtype=np.float64
    )
    ms_after = np.asarray(
        [
            record.get("ms_after", int(num_samples[index]))
            for index, record in enumerate(records)
        ],
        dtype=np.float64,
    )
    n_spikes_used = np.asarray(
        [record.get("n_spikes_used", -1) for record in records], dtype=np.int64
    )
    grid_distance, is_multichannel = classify_waveform_channel_construction(
        bioD["node_positions"]
    )
    if grid_distance.shape != unit_ids.shape:
        raise ValueError(
            "node_positions and waveform unit count do not match: "
            f"{grid_distance.size} != {unit_ids.size}"
        )

    bioD["raw_mean_templates"] = waveform_2d
    bioD["waveform_num_samples"] = num_samples
    bioD["waveform_unit_ids"] = _integer_if_possible(unit_ids)
    bioD["waveform_channel_ids"] = channel_ids
    bioD["waveform_ms_before"] = ms_before
    bioD["waveform_ms_after"] = ms_after
    bioD["waveform_n_spikes_used"] = n_spikes_used
    bioD["waveform_grid_distance"] = grid_distance
    bioD["waveform_is_multichannel"] = is_multichannel

    bioMD["bioexp_schema_version"] = BIOEXP_SCHEMA_VERSION
    bioMD["waveforms_available"] = True
    bioMD["waveform_schema"] = {
        "version": 2,
        "template_record": "raw_mean_templates",
        "neuron_axis": "frequency-sorted; aligned with MEA_idx and spike_key",
        "sample_padding": "NaN beyond waveform_num_samples",
        "time_axis": "linspace(-waveform_ms_before, waveform_ms_after, "
                     "waveform_num_samples, endpoint=False)",
        "channel_construction_record": "waveform_is_multichannel",
        "grid_distance_record": "waveform_grid_distance",
        "grid_pitch_xy": [WAVEFORM_GRID_PITCH, WAVEFORM_GRID_PITCH],
        "grid_origin_xy": WAVEFORM_GRID_ORIGIN.tolist(),
        "multichannel_rule": "distance_from_nearest_grid_point > 1",
        "multichannel_distance_threshold": (
            WAVEFORM_MULTICHANNEL_DISTANCE_THRESHOLD
        ),
    }
    print(
        "raw mean waveforms: units=%d, stored shape=%s, single-channel=%d, "
        "multi-channel=%d"
        % (
            len(unit_ids),
            waveform_2d.shape,
            int(np.sum(~is_multichannel)),
            int(np.sum(is_multichannel)),
        )
    )


def load_metrics_curated(args, mea_idx_order, spike_key_order):
    """Load metrics spreadsheet; filter and reorder rows to match output neuron order."""
    df, metrics_name = _read_metrics_frame(args)

    mea_keys = [_mea_match_key(v) for v in df["MEA_idx"].values]
    if len(mea_keys) != len(set(mea_keys)):
        raise ValueError(f"duplicate MEA_idx rows in {metrics_name}")

    row_by_mea = {_k: i for i, _k in enumerate(mea_keys)}
    order_idx = []
    for mea_id in np.asarray(mea_idx_order).ravel():
        key = _mea_match_key(mea_id)
        assert key in row_by_mea, (
            f"MEA_idx {mea_id!r} (key={key!r}) not found in {metrics_name}"
        )
        order_idx.append(row_by_mea[key])

    df_ord = df.iloc[order_idx].reset_index(drop=True)
    col_names = [str(c) for c in df_ord.columns]
    metrics_2d = df_ord.to_numpy()
    if metrics_2d.dtype.hasobject:
        # A heterogeneous DataFrame becomes an object array, which would
        # require pickle.  Store it as schema-safe Unicode instead.  Numeric
        # metric consumers select a column and explicitly cast it to float.
        schema_safe_df = df_ord.where(pd.notna(df_ord), np.nan)
        metrics_2d = schema_safe_df.astype(str).to_numpy(dtype=str)
    mea_idx_order = _integer_if_possible(df_ord["MEA_idx"].to_numpy())

    n_match = len(order_idx)
    n_sheet = len(df)
    print(
        "%s: kept %d/%d rows, shape=%s, cols=%d"
        % (metrics_name, n_match, n_sheet, metrics_2d.shape, len(col_names))
    )
    if args.verb > 1:
        print("  columns:", col_names)
    return metrics_2d, col_names, mea_idx_order


#...!...!....................
def unroll_bioexp(rawD, bioMD, args):
    pmd = bioMD['bioexp']
    sel = bioMD['data_selector']
    frLo, frHi = sel['freq_range']
    print('frLo, frHi', frLo, frHi)
    assert frLo < frHi
    chanFreq = np.asarray(rawD['chanFreq'], dtype=float)
    # vectorized boolean mask for channels within (frLo, frHi) range
    freqMask = (chanFreq >= frLo) & (chanFreq <= frHi)
    sel['num_drop_neur_lo_hi_freq'] = [int(np.sum(chanFreq < frLo)), int(np.sum(chanFreq > frHi))]
    print('freqMask all=%d , passed=%d' % (freqMask.shape[0], np.sum(freqMask)))
    #print(sel);aaa
    # --- drop channles out of freq range
    chanFreq = rawD['chanFreq'][freqMask]
    MEA_idx = rawD['MEA_idx'][freqMask]
    spike_key = rawD['spike_key'][freqMask]

    # .... REMAP MATRICES TO FREQUENCY-SORTED ORDER (PRIMARY INDEX)
    neur_freqIdx = np.argsort(chanFreq)  # indices that sort chanFreq by value
    neur_revFreqIdx = np.empty(len(neur_freqIdx), dtype=int)  # natural_index -> freq_sorted_position
    neur_revFreqIdx[neur_freqIdx] = np.arange(len(neur_freqIdx))

    #--- reorder channles by frequency
    chanFreq = chanFreq[neur_freqIdx]
    MEA_idx = MEA_idx[neur_freqIdx]
    spike_key = spike_key[neur_freqIdx]
    print('chanFreq', chanFreq[:5], '...', chanFreq[-5:], 'Hz')

    # create spike matrix: rows=time bins, cols=accepted channels
    ntime = pmd['num_time_bin']
    nchan = spike_key.shape[0]
    spikes2D = np.zeros((ntime, nchan), dtype=np.int32)
    spikeT = rawD['spikeT']
    for ic, key in enumerate(spike_key):
        tV = spikeT[key]
        if len(tV) == 0:
            continue
        # tV holds time-bin indices where this channel fired one or more spikes
        # bincount returns a length-ntime vector with spike counts per bin (zeros elsewhere)
        # minlength=ntime guarantees the vector spans the full recording duration
        cnt = np.bincount(tV, minlength=ntime)
        spikes2D[:, ic] = cnt.astype(np.int32)
    print('spikes2D shape', spikes2D.shape)

    # keep handy in meta for downstream
    sel['num_chan'] = nchan
    sel['max_spike_per_bin'] = int(np.max(spikes2D))

    Y_uchar = np.clip(spikes2D, 0, 255).astype(np.uint8)
    spikeD = {'spikes': Y_uchar,
              'single_rates': chanFreq
              }

    bioD = {}
    bioD['neur_freqIdx'] = neur_freqIdx
    bioD['neur_revFreqIdx'] = neur_revFreqIdx
    bioD['single_rates'] = np.asarray(chanFreq, dtype=np.float64)

    metrics_2d, metrics_cols, MEA_idx = load_metrics_curated(args, MEA_idx, spike_key)
    bioD['MEA_idx'] = MEA_idx
    bioD['spike_key'] = spike_key
    bioD['metrics_curated'] = metrics_2d
    bioMD['metrics_curated_columns'] = metrics_cols
    metrics_col_map = {str(c): i for i, c in enumerate(metrics_cols)}
    assert "loc_x" in metrics_col_map and "loc_y" in metrics_col_map, (
        f"metrics_curated must contain loc_x and loc_y columns; have {metrics_cols}"
    )
    loc_x = metrics_2d[:, metrics_col_map["loc_x"]].astype(np.float64)
    loc_y = metrics_2d[:, metrics_col_map["loc_y"]].astype(np.float64)
    bioD['node_positions'] = np.column_stack([loc_x, loc_y])

    #.... compute neural statistics for spikeMD
    num_neurons = nchan
    avg_rate = float(np.mean(chanFreq))
    std_rate = float(np.std(chanFreq))
    median_rate = float(np.median(chanFreq))
    min_rate = float(np.min(chanFreq))
    max_rate = float(np.max(chanFreq))

    # Compute Fano factor (variance/mean) for each neuron
    mean_counts_per_bin = np.mean(spikes2D, axis=0)
    spike_variance = np.var(spikes2D, axis=0)
    fano_factor = np.divide(spike_variance, mean_counts_per_bin, out=np.zeros_like(spike_variance), where=mean_counts_per_bin != 0)
    avg_fano = float(np.mean(fano_factor))
    std_fano = float(np.std(fano_factor))

    # Print summary statistics
    recording_minutes = pmd['max_time'] / 60.0
    print('Neural Statistics Summary: %.2f minutes of recording available' % recording_minutes)
    print('num neurons: %d, Avg Rate= %.2f±%.2f Hz, Avg Fano=%.2f±%.2f' % (num_neurons, avg_rate, std_rate, avg_fano, std_fano))
    print('Median rate  %.2f Hz' % median_rate)

    #.... extract spikeMD for fitter
    spikeMD = {'time_step_sec': 1. / pmd['sampling_freq'],
               'provenance': {'experiment_name': bioMD['short_name']},
               'poisson_eta_clip': 5,  # expected by fitter
               'data_type': 'bioExp', 'num_neurons': num_neurons}

    bioMD['rate_summary'] = {
        'avg_spike_rate': avg_rate,
        'std_spike_rate': std_rate,
        'avg_fano_factor': avg_fano,
        'std_fano_factor': std_fano,
        'median_spike_rate': median_rate,
        'min_spike_rate': min_rate,
        'max_spike_rate': max_rate
    }
    return bioD, spikeD, spikeMD


#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":

    args = commandline_parser()
    np.set_printoptions(precision=3)
    bioMD = buildBioMeta(args)

    # read raw data
    #rawD=read_spike_dict(bioMD,args)
    rawD = read_spike_npy(bioMD, args)
    templates, waveform_source = load_raw_mean_templates(args)

    #.... filter & unroll data
    bioD, spikeD, spikeMD = unroll_bioexp(rawD, bioMD, args)
    add_raw_mean_waveforms(bioD, bioMD, templates, waveform_source)

    #...... WRITE   OUTPUT .........
    outFt = os.path.join(args.dataPath, bioMD['short_name'] + '.bioExp.npz')
    write_data_npz(bioD, outFt, metaD=json_safe_metadata(bioMD))
    print("Saved NPZ:", os.path.abspath(outFt))
    if args.verb > 2:
        print('\n bioD:', sorted(bioD))
        pprint(bioMD)

    outFs = outFt.replace('.bioExp.', '.spikes.')
    write_data_npz(spikeD, outFs, metaD=json_safe_metadata(spikeMD))
    print("Saved NPZ:", os.path.abspath(outFs))
    if args.verb > 2:
        print('\nspikeD:', sorted(spikeD))
        pprint(spikeMD)

    print("\nNext step command:")
    print('   ./view_bioexp.py  --dataPath $dataPath  --dataName   %s  -p  a b c  -T 0 3550  ' % (bioMD['short_name']))

    print("\nRun one PRISM-EM fit:")
    print("  ./fitPrismEM.sh --basePath $basePath --dataName %s --num_states 2 --num_em_iters 4 --m_epochs 16 --time_range_sec 0 80" % bioMD['short_name'])
    print("\nRun the full PRISM-EM/FDR bags pipeline (configure dataset selection in the script first):")
    print("  ./big_fit_bags.sh")

    print('    dataPath=%s' % args.dataPath)
