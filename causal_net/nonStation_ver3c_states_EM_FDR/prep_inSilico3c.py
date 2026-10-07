#!/usr/bin/env python3
"""
Preprocessing pipeline for in-silico neural data: NetPyNE biophysical network
simulations of synthetic neuronal networks (e.g. Kaustubh's netpyne recordings).

Input: a single NetPyNE sim-data pickle at  <expPath>/<sessionName>.pkl , with
the standard NetPyNE sim.saveData() layout:
  netpyne_version, netpyne_changeset
  net:      {params, cells, pops}
  simConfig: {duration, dt, num_excite, num_inhib, ...}
  simData:   {spkt, spkid, popRates, avgRate, ...}

The pickle references a handful of NetPyNE/NEURON classes deep under
net['params'] (cell morphology, mechanism templates). Rather than requiring
a full netpyne+NEURON install just to load the
file, load_netpyne_pickle() below tolerates unresolvable classes by swapping
in an inert placeholder, since net['cells'], net['pops'], simData and
simConfig are already plain dicts/lists once unpickled.

Outputs (written to --dataPath, mirroring gen_nonStationarySpikes3c.py):
  <shortName>.spikes.npz     — binned spike counts + single_rates, for training
  <shortName>.prismTruth.npz — ground-truth network structure, for evaluation
                                only. node_positions/node_is_inhibitory/gid are
                                stored as arrays; the per-neuron synapse list
                                (preGid, weight, delay, synMech) is stored as a
                                compact nested dict in the JSON metadata (this
                                is raw biophysical synaptic structure, not the
                                fitter's log-rate A matrix, so it is not
                                magnitude-comparable to a fitted A_hat).
                                postsynaptic_dynamics metadata preserves the
                                synapse parameters, cell-model rules, per-neuron
                                model tags, and simulation timestep. These
                                describe response kinetics, not a measured or
                                guaranteed postsynaptic firing latency.

Usage:
    ./prep_inSilico3c.py --expPath /path/to/raw/data --sessionName trial_159_baseline_rerun_3600s_data_trim5s \
        --freqRange 0.3 50 --dataPath /path/to/spikesData/ --shortName silico_20260831_r159
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import os
import pickle
import hashlib
import argparse
from pprint import pprint

import numpy as np

from toolbox.Util_NumpyIOv2 import json_safe_metadata, write_data_npz


#...!...!..................
class _InertPlaceholder(dict):
    """Standin for any pickled class this script cannot (and need not) import.

    Deep inside net['params'] NetPyNE stores cell-model/mechanism objects that
    require the real netpyne/NEURON packages to reconstruct. Plain model
    parameters can still be exported; unavailable objects are explicitly
    marked as missing in the postsynaptic-dynamics metadata.
    """

    def __init__(self, *args, **kwargs):
        dict.__init__(self)

    def __setstate__(self, state):
        self._state = state

    def append(self, item):
        pass

    def extend(self, items):
        pass


class _PlainParameterDict(dict):
    """Fallback for known NetPyNE mappings whose pickle state is their data."""

    def __setstate__(self, state):
        if not isinstance(state, dict):
            raise TypeError("expected dictionary state for NetPyNE parameters")
        self.update(state)


class _TolerantUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            if ((module == "netpyne.specs.dicts" and name in ("Dict", "ODict"))
                    or (module == "netpyne.specs.netParams"
                        and name in ("CellParams", "SynMechParams"))):
                return _PlainParameterDict
            return _InertPlaceholder


#...!...!..................
def load_netpyne_pickle(inpF):
    assert os.path.exists(inpF), f"missing input pickle: {inpF}"
    with open(inpF, "rb") as f:
        raw = _TolerantUnpickler(f).load()
    for key in ("net", "simConfig", "simData"):
        assert key in raw, f"{inpF} is missing top-level key {key!r}"
    return raw


#...!...!..................
def extract_postsynaptic_dynamics(raw):
    """Preserve model parameters without inventing a spike-latency estimate.

    Keep native NetPyNE names/values, including custom mechanism parameters.
    Missing/unreadable fields are null and listed explicitly, never replaced
    by NEURON defaults. Mechanism implementations and runtime states are not
    reconstructed by this export.
    """
    unavailable = []

    def plain(value, path):
        if isinstance(value, _InertPlaceholder):
            unavailable.append(path)
            return None
        if isinstance(value, np.ndarray):
            return plain(value.tolist(), path)
        if isinstance(value, np.generic):
            return plain(value.item(), path)
        if isinstance(value, dict):
            return {str(k): plain(v, path + "." + str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v, "%s[%d]" % (path, i)) for i, v in enumerate(value)]
        if isinstance(value, float) and not np.isfinite(value):
            unavailable.append(path)
            return None
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        unavailable.append(path)
        return None

    params = raw["net"].get("params")

    def parameter(name):
        path = "net.params." + name
        if (not isinstance(params, dict) or isinstance(params, _InertPlaceholder)
                or name not in params or params[name] is None):
            unavailable.append(path)
            return None
        return plain(params[name], path)

    dynamics = {
        "schema_version": 1,
        "source": "raw NetPyNE pickle: net.params, net.cells[].tags, simConfig.dt",
        "sim_dt_ms": float(raw["simConfig"]["dt"]),
        "synMechParams": parameter("synMechParams"),
        "cellParams": parameter("cellParams"),
        "defaultThreshold_mV": parameter("defaultThreshold"),
        "units": "Native NEURON mechanism units; synaptic tau1, tau2, tau_rec, "
                 "and tau_facil are in ms for the supplied mechanisms. "
                 "Other/custom parameters retain their mechanism-specific units.",
        "interpretation": "Connectivity delay_ms is presynaptic event delivery "
                          "delay. Synaptic rise/decay and recovery/facilitation "
                          "constants describe response and plasticity kinetics, "
                          "not postsynaptic firing latency. Firing depends on "
                          "membrane state, other inputs, and mechanism dynamics; "
                          "no finite maximum firing latency is established here.",
        "model_scope": "cellParams are original cell rules, not reconstructed "
                       "per-cell runtime states. node_model_tags uses the same "
                       "frequency-sorted axis as node_gid and records rule "
                       "selection tags when available. Synapse code c maps to "
                       "synMechParams[synMech_legend[c]]. Custom mechanism code "
                       "and voltage/current traces are not included.",
        "unavailable_fields": unavailable,
    }
    tags_by_gid = {}
    for cell in raw["net"]["cells"]:
        gid = int(cell["gid"])
        tags_by_gid[gid] = plain(
            cell["tags"],
            "net.cells[gid=%d].tags" % gid)
    return dynamics, tags_by_gid


#...!...!..................
def commandline_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verb", type=int, help="increase debug verbosity", default=1)
    parser.add_argument("--expPath", required=True, help="dir holding the raw in-silico *.pkl")
    parser.add_argument("--sessionName", required=True, help="pickle base name, file is <expPath>/<sessionName>.pkl")
    parser.add_argument("--dataPath", default="/pscratch/sd/b/balewski/2025_causalNet_tmp/", help="head dir for any further data processing")
    parser.add_argument("--shortName", default=None, help="(optional) output file base name")

    parser.add_argument("--samp_freq", default=100, type=int, help="sets binning of time axis (Hz)")
    parser.add_argument("--freqRange", default=[1., 50], type=float, nargs=2, help="keep only neurons with mean rate inside [lo, hi] Hz")

    args = parser.parse_args()

    for arg in vars(args):
        print("myArgs:", arg, getattr(args, arg))

    return args


#...!...!..................
def buildInSilicoMeta(args):
    pd = {}  # payload
    pd["raw_insilico_path"] = args.expPath
    pd["session_name"] = args.sessionName

    sel = {"freq_range": args.freqRange, "samp_freq": args.samp_freq}
    md = {"insilico": pd, "data_selector": sel}
    myHN = hashlib.md5(os.urandom(32)).hexdigest()[:6]
    md["hash"] = myHN
    md["short_name"] = args.shortName if args.shortName is not None else "silico-%s" % myHN

    if args.verb > 1:
        print("\nISMD:")
        pprint(md)
    return md


#...!...!..................
def read_netpyne_network(md, args):
    """Extract cells/pops/spikes/sim-config bookkeeping from the raw pickle."""
    pmd = md["insilico"]
    inpF = os.path.join(args.expPath, args.sessionName + ".pkl")
    print("inpF:", inpF)
    raw = load_netpyne_pickle(inpF)
    postsynaptic_dynamics, model_tags_by_gid = extract_postsynaptic_dynamics(raw)

    pmd["netpyne_version"] = raw.get("netpyne_version")
    simConfig = raw["simConfig"]
    pmd["sim_dt_ms"] = float(simConfig["dt"])
    pmd["sim_duration_ms"] = float(simConfig["duration"])
    pmd["sim_filename"] = simConfig.get("filename")
    duration_sec = pmd["sim_duration_ms"] / 1000.0
    pmd["duration_sec"] = duration_sec

    pops = raw["net"]["pops"]
    assert "E" in pops and "I" in pops, f"expected pops 'E' and 'I', got {list(pops)}"
    excite_gids = np.asarray(sorted(int(g) for g in pops["E"]["cellGids"]))
    inhib_gids = np.asarray(sorted(int(g) for g in pops["I"]["cellGids"]))
    num_excite_total = len(excite_gids)
    num_inhib_total = len(inhib_gids)

    cells = raw["net"]["cells"]
    num_neurons_total = len(cells)
    assert num_neurons_total == num_excite_total + num_inhib_total, (
        f"cell count {num_neurons_total} != E({num_excite_total}) + I({num_inhib_total})"
    )
    pmd["num_neurons_total"] = num_neurons_total
    pmd["num_excite_total"] = num_excite_total
    pmd["num_inhib_total"] = num_inhib_total
    print("Initial neuron counts: total=%d, E=%d, I=%d" % (
        num_neurons_total, num_excite_total, num_inhib_total))

    gid2idx = {int(c["gid"]): i for i, c in enumerate(cells)}
    node_pos_all = np.zeros((num_neurons_total, 2), dtype=np.float64)
    conns_by_gid = {}
    for gid, idx in gid2idx.items():
        tags = cells[idx]["tags"]
        node_pos_all[gid] = [float(tags["x"]), float(tags["y"])]
        conns_by_gid[gid] = cells[idx]["conns"]

    # .... spike times: bin (ms) -> integer time-bin at samp_freq
    spkt = np.asarray(raw["simData"]["spkt"], dtype=np.float64)
    spkid = np.asarray(raw["simData"]["spkid"], dtype=np.int64)
    assert spkt.shape == spkid.shape
    tbin = np.floor(spkt / 1000.0 * args.samp_freq).astype(np.int64)
    num_time_bin = int(np.ceil(duration_sec * args.samp_freq))
    keep = (tbin >= 0) & (tbin < num_time_bin)
    if args.verb > 0 and np.sum(~keep) > 0:
        print("dropping %d/%d spikes falling outside [0, duration)" % (np.sum(~keep), tbin.size))
    tbin, spkid = tbin[keep], spkid[keep]

    order = np.argsort(spkid, kind="stable")
    spkid_sorted, tbin_sorted = spkid[order], tbin[order]
    boundaries = np.searchsorted(spkid_sorted, np.arange(num_neurons_total + 1))

    spikeCntL = np.zeros(num_neurons_total, dtype=np.int64)
    spikeT = {}
    for gid in range(num_neurons_total):
        lo, hi = boundaries[gid], boundaries[gid + 1]
        spikeT[gid] = tbin_sorted[lo:hi]
        spikeCntL[gid] = hi - lo
    chanFreq_all = spikeCntL / duration_sec

    pmd["num_time_bin"] = num_time_bin
    pmd["time_step_sec"] = 1.0 / args.samp_freq

    rawD = {
        "spikeT": spikeT,
        "chanFreq_all": chanFreq_all,
        "node_pos_all": node_pos_all,
        "conns_by_gid": conns_by_gid,
        "excite_gids": excite_gids,
        "inhib_gids": inhib_gids,
        "num_time_bin": num_time_bin,
        "postsynaptic_dynamics": postsynaptic_dynamics,
        "model_tags_by_gid": model_tags_by_gid,
    }
    return rawD


#...!...!..................
def unroll_inSilico(rawD, insilicoMD, args):
    """Apply the freqRange neuron filter and build the training + truth payloads.

    Matching prep_bioexp3c.py convention: the primary neuron index in spikeD is
    frequency-sorted (ascending), NOT Dale-contiguous. E/I identity per position
    is preserved via node_is_inhibitory (a boolean array, no contiguity assumed).
    neur_freqIdx/neur_revFreqIdx and node_gid are stored so downstream code can
    recover the natural (Dale-contiguous, GID-based) order from the raw .pkl.
    """
    pmd = insilicoMD["insilico"]
    sel = insilicoMD["data_selector"]
    frLo, frHi = sel["freq_range"]
    print("frLo, frHi", frLo, frHi)
    assert frLo < frHi

    chanFreq_all = rawD["chanFreq_all"]

    def _survivors(gids):
        freqs = chanFreq_all[gids]
        mask = (freqs >= frLo) & (freqs <= frHi)
        return gids[mask]

    excite_kept = _survivors(rawD["excite_gids"])
    inhib_kept = _survivors(rawD["inhib_gids"])
    all_gids = np.concatenate([rawD["excite_gids"], rawD["inhib_gids"]])
    all_freqs = chanFreq_all[all_gids]
    sel["num_drop_neur_lo_hi_freq"] = [int(np.sum(all_freqs < frLo)), int(np.sum(all_freqs > frHi))]

    # .... natural (Dale-contiguous) order of surviving neurons, pre-frequency-sort
    node_order_natural = np.concatenate([excite_kept, inhib_kept])
    num_excite = len(excite_kept)
    num_neurons = len(node_order_natural)
    print("freqMask: E kept=%d/%d, I kept=%d/%d, total kept=%d" % (
        num_excite, len(rawD["excite_gids"]), len(inhib_kept), len(rawD["inhib_gids"]), num_neurons))
    assert num_excite > 0 and num_neurons - num_excite > 0, "freqRange filter removed an entire E or I population"
    med_freq_excite = float(np.median(chanFreq_all[excite_kept]))
    med_freq_inhib = float(np.median(chanFreq_all[inhib_kept]))
    print("Median accepted rate: E=%.2f Hz, I=%.2f Hz" % (med_freq_excite, med_freq_inhib))

    is_inhib_natural = np.concatenate([
        np.zeros(num_excite, dtype=bool), np.ones(num_neurons - num_excite, dtype=bool)])
    chanFreq_natural = chanFreq_all[node_order_natural]

    # .... REMAP TO FREQUENCY-SORTED ORDER (PRIMARY INDEX), mirroring prep_bioexp3c.py
    neur_freqIdx = np.argsort(chanFreq_natural)  # freq-sorted position -> natural position
    neur_revFreqIdx = np.empty(num_neurons, dtype=int)  # natural position -> freq-sorted position
    neur_revFreqIdx[neur_freqIdx] = np.arange(num_neurons)

    node_order = node_order_natural[neur_freqIdx]
    node_is_inhibitory = is_inhib_natural[neur_freqIdx].astype(np.int32)

    old2new = {int(gid): j for j, gid in enumerate(node_order)}
    chanFreq = chanFreq_all[node_order]
    print("chanFreq", chanFreq[:5], "...", chanFreq[-5:], "Hz")

    # .... training payload: binned spike-count matrix
    ntime = rawD["num_time_bin"]
    spikes2D = np.zeros((ntime, num_neurons), dtype=np.int32)
    spikeT = rawD["spikeT"]
    for jnew, gid in enumerate(node_order):
        tV = spikeT[int(gid)]
        if len(tV) == 0:
            continue
        spikes2D[:, jnew] = np.bincount(tV, minlength=ntime).astype(np.int32)
    print("spikes2D shape", spikes2D.shape)

    sel["num_chan"] = num_neurons
    sel["max_spike_per_bin"] = int(np.max(spikes2D))

    Y_uchar = np.clip(spikes2D, 0, 255).astype(np.uint8)
    spikeD = {"spikes": Y_uchar, "single_rates": chanFreq}

    # .... ground-truth payload: node geometry/type as arrays, synapse list as a
    #      compact nested dict (post_idx -> [[pre_idx, weight, delay, synCode], ...])
    node_positions = rawD["node_pos_all"][node_order]
    node_gid = node_order.astype(np.int32)

    conns_by_gid = rawD["conns_by_gid"]
    syn_vocab = {}
    connectivity = {}
    n_edge_kept, n_edge_drop_filtered_pre = 0, 0
    for jnew, gid in enumerate(node_order):
        edges = []
        for conn in conns_by_gid[int(gid)]:
            pre_gid = int(conn["preGid"])
            if pre_gid not in old2new:
                n_edge_drop_filtered_pre += 1
                continue
            synMech = str(conn["synMech"])
            if synMech not in syn_vocab:
                syn_vocab[synMech] = len(syn_vocab)
            edges.append([old2new[pre_gid], round(float(conn["weight"]), 6),
                          round(float(conn["delay"]), 6), syn_vocab[synMech]])
            n_edge_kept += 1
        if edges:
            connectivity[str(jnew)] = edges

    syn_legend = [None] * len(syn_vocab)
    for name, code in syn_vocab.items():
        syn_legend[code] = name

    postsynaptic_dynamics = dict(rawD["postsynaptic_dynamics"])
    postsynaptic_dynamics["node_model_tags"] = [
        rawD["model_tags_by_gid"][int(gid)] for gid in node_order]

    truthD = {
        "node_positions": node_positions,
        "node_is_inhibitory": node_is_inhibitory,
        "node_gid": node_gid,
        "neur_freqIdx": neur_freqIdx,
        "neur_revFreqIdx": neur_revFreqIdx,
    }
    dale_conf = {"num_neurons": num_neurons, "num_excite": num_excite,
                 "num_inhib": num_neurons - num_excite}
    truthMD = {
        "short_name": insilicoMD["short_name"],
        "data_type": "silico",
        "num_neurons": num_neurons,
        "data_selector": sel,
        "dale_conf": dale_conf,
        "neuron_axis": "frequency-sorted (ascending); NOT Dale-contiguous -- "
                        "use node_is_inhibitory for per-neuron E/I type, "
                        "node_gid + neur_freqIdx/neur_revFreqIdx to recover the "
                        "natural (Dale-contiguous, GID-based) order from the raw .pkl",
        "connectivity": connectivity,
        "synMech_legend": syn_legend,
        "postsynaptic_dynamics": postsynaptic_dynamics,
        "connectivity_schema": "post_local_idx (str key) -> list of "
                                "[pre_local_idx, weight, delay_ms, synMech_code]; "
                                "raw NetPyNE synaptic weights, NOT the fitter's "
                                "log-rate A matrix -- topology/sign truth only",
        "num_edges_kept": n_edge_kept,
        "num_edges_dropped_filtered_presyn": n_edge_drop_filtered_pre,
        "provenance": {"sim_source": pmd["session_name"], "state_transition_file": insilicoMD["short_name"]},
    }

    # .... neural statistics for spikeMD / console summary
    avg_rate = float(np.mean(chanFreq))
    std_rate = float(np.std(chanFreq))
    mean_counts_per_bin = np.mean(spikes2D, axis=0)
    spike_variance = np.var(spikes2D, axis=0)
    fano_factor = np.divide(spike_variance, mean_counts_per_bin,
                             out=np.zeros_like(spike_variance), where=mean_counts_per_bin != 0)
    print("Neural Statistics Summary: %.2f minutes of recording, N=%d (E=%d, I=%d)" % (
        pmd["duration_sec"] / 60.0, num_neurons, num_excite, num_neurons - num_excite))
    print("Avg Rate= %.2f±%.2f Hz, Avg Fano=%.2f±%.2f" % (avg_rate, std_rate, np.mean(fano_factor), np.std(fano_factor)))
    print("kept edges=%d, dropped (presyn filtered out)=%d" % (n_edge_kept, n_edge_drop_filtered_pre))

    spikeMD = {
        "time_step_sec": pmd["time_step_sec"],
        "provenance": {"sim_source": pmd["session_name"], "state_transition_file": insilicoMD["short_name"]},
        "poisson_eta_clip": 5,  # expected by fitter
        "data_type": "silico",
        "num_neurons": num_neurons,
    }

    truthMD["rate_summary"] = {
        "avg_spike_rate": avg_rate,
        "std_spike_rate": std_rate,
        "median_spike_rate": float(np.median(chanFreq)),
        "min_spike_rate": float(np.min(chanFreq)),
        "max_spike_rate": float(np.max(chanFreq)),
        "avg_fano_factor": float(np.mean(fano_factor)),
        "std_fano_factor": float(np.std(fano_factor)),
    }
    return spikeD, spikeMD, truthD, truthMD


#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":

    args = commandline_parser()
    np.set_printoptions(precision=3)
    insilicoMD = buildInSilicoMeta(args)

    rawD = read_netpyne_network(insilicoMD, args)
    spikeD, spikeMD, truthD, truthMD = unroll_inSilico(rawD, insilicoMD, args)

    #...... WRITE   OUTPUT .........
    outFs = os.path.join(args.dataPath, insilicoMD["short_name"] + ".spikes.npz")
    write_data_npz(spikeD, outFs, metaD=json_safe_metadata(spikeMD))
    print("Saved NPZ:", os.path.abspath(outFs))
    if args.verb > 2:
        print("\nspikeD:", sorted(spikeD))
        pprint(spikeMD)

    outFt = outFs.replace(".spikes.", ".prismTruth.")
    write_data_npz(truthD, outFt, metaD=json_safe_metadata(truthMD))
    print("Saved NPZ:", os.path.abspath(outFt))
    if args.verb > 2:
        print("\ntruthD:", sorted(truthD))
        pprint(truthMD)

    print("\nRun one PRISM-EM fit:")
    print("  ./fitPrismEM.sh --basePath $basePath --dataName %s --num_states 2 --num_em_iters 4 --m_epochs 16 --time_range_sec 0 80" % insilicoMD["short_name"])
    print("\nRun the full PRISM-EM/FDR bags pipeline (configure dataset selection in the script first):")
    print("  ./big_fit_bags.sh")

    print("    dataPath=%s" % args.dataPath)

    print("\nView the dataset:")
    print("  ./view_inSilico.py --dataPath $dataPath --dataName %s  -p a b c -T 5 35 -X --time_rebin2 5" % insilicoMD["short_name"])
