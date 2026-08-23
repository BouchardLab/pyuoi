#!/usr/bin/env python3
"""Plot selected raw mean spike waveforms and their node locations."""

import argparse
from collections import Counter
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot selected unit waveforms in a three-column canvas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rawPath",
        default="/global/cfs/cdirs/m2043/causal_inference/KCL_experiment/260729/M07420/Network/000018/well005/",
        help="Directory containing raw_mean_templates.npy and metrics_curated.xlsx",
    )
    parser.add_argument(
        "--metricsPath",
        default=None,
        help=(
            "Path to metrics_curated.xlsx; by default it is read from rawPath"
        ),
    )
    parser.add_argument(
        "--gridPitch",
        type=float,
        default=17.5,
        help="Electrode-grid spacing in the loc_x/loc_y coordinate units",
    )
    parser.add_argument(
        "--gridOrigin",
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        metavar=("X", "Y"),
        help="Electrode-grid origin",
    )
    parser.add_argument(
        "--gridTolerance",
        type=float,
        default=1.0,
        help=(
            "Maximum distance from the nearest grid point for a unit to be "
            "classified as grid-aligned"
        ),
    )
    parser.add_argument(
        "--multiChanSeed",
        type=int,
        default=42,
        help="Random seed used to select waveforms for multiChan.png",
    )
    parser.add_argument(
        "--index",
        type=int,
        nargs="+",
        default=[1, 51, 101, 151, 201, 251, 301, 351],
        help="Zero-based indices into the numerically sorted unit-ID list",
    )
    parser.add_argument(
        "--outPath",
        default="out/",
        help="Directory in which to save the output PNG",
    )
    return parser.parse_args()


def load_templates(raw_path):
    input_file = os.path.join(raw_path, "raw_mean_templates.npy")
    if not os.path.isfile(input_file):
        raise FileNotFoundError("Missing input file: %s" % input_file)

    container = np.load(input_file, allow_pickle=True)
    if container.shape != () or container.dtype != object:
        raise ValueError(
            "%s must contain a scalar NumPy object holding a dictionary" % input_file
        )

    templates = container.item()
    if not isinstance(templates, dict):
        raise TypeError("Expected a dictionary in %s" % input_file)
    return templates, input_file


def get_record(templates, unit_id):
    """Accept files whose dictionary keys are either strings or integers."""
    if str(unit_id) in templates:
        return templates[str(unit_id)]
    if unit_id in templates:
        return templates[unit_id]
    return None


def mea_match_key(value):
    """Canonical key for matching template unit IDs to spreadsheet MEA_idx."""
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            raise ValueError("MEA_idx is NaN in metrics spreadsheet")
        number = float(value)
        return str(int(number)) if number == int(number) else str(number)

    text = str(value).strip()
    try:
        number = float(text)
        return str(int(number)) if number == int(number) else str(number)
    except ValueError:
        return text


def selected_units_from_indices(templates, indices):
    available_units = sorted(int(unit_id) for unit_id in templates)
    invalid = [index for index in indices if index < 0 or index >= len(available_units)]
    if invalid:
        raise IndexError(
            "Requested index(es) outside the available range 0..%d: %s"
            % (len(available_units) - 1, " ".join(map(str, invalid)))
        )
    return available_units, [available_units[index] for index in indices]


def channel_unit_multiplicities(templates, available_units):
    units_per_channel = Counter()
    channel_by_unit = {}
    for unit_id in available_units:
        record = get_record(templates, unit_id)
        channel_id = record["primary_channel"]
        channel_by_unit[unit_id] = channel_id
        units_per_channel[channel_id] += 1

    other_units_by_unit = {
        unit_id: units_per_channel[channel_id] - 1
        for unit_id, channel_id in channel_by_unit.items()
    }
    return Counter(units_per_channel.values()), other_units_by_unit


def load_node_locations(raw_path, metrics_path=None):
    metrics_file = metrics_path or os.path.join(raw_path, "metrics_curated.xlsx")
    if not os.path.isfile(metrics_file):
        raise FileNotFoundError("Missing metrics spreadsheet: %s" % metrics_file)

    frame = pd.read_excel(metrics_file, engine="openpyxl")
    if frame.shape[1] == 0:
        raise ValueError("Metrics spreadsheet has no columns: %s" % metrics_file)

    columns = list(frame.columns)
    columns[0] = "MEA_idx"
    frame.columns = columns
    required = ["MEA_idx", "loc_x", "loc_y"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            "Missing required metrics column(s) in %s: %s"
            % (metrics_file, ", ".join(missing))
        )

    locations = {}
    for mea_id, loc_x, loc_y in frame[required].itertuples(index=False, name=None):
        key = mea_match_key(mea_id)
        if key in locations:
            raise ValueError("Duplicate MEA_idx %r in %s" % (mea_id, metrics_file))
        coordinates = np.asarray([loc_x, loc_y], dtype=float)
        if not np.all(np.isfinite(coordinates)):
            raise ValueError(
                "Non-finite location for MEA_idx %r in %s" % (mea_id, metrics_file)
            )
        locations[key] = tuple(coordinates)

    return locations, metrics_file


def distances_from_electrode_grid(locations, pitch, origin):
    """Return nearest grid coordinates, offsets, and distances for all units."""
    if not np.isfinite(pitch) or pitch <= 0.0:
        raise ValueError("grid pitch must be a finite positive number")

    origin = np.asarray(origin, dtype=float)
    if origin.shape != (2,) or not np.all(np.isfinite(origin)):
        raise ValueError("grid origin must contain two finite coordinates")

    grid_offsets = {}
    for unit_id, coordinates in locations.items():
        coordinates = np.asarray(coordinates, dtype=float)
        nearest_grid = origin + np.rint((coordinates - origin) / pitch) * pitch
        delta = coordinates - nearest_grid
        grid_offsets[unit_id] = {
            "nearest_grid": tuple(nearest_grid),
            "delta": tuple(delta),
            "distance": float(np.linalg.norm(delta)),
        }
    return grid_offsets


def waveform_time_ms(record, num_samples):
    ms_before = float(record.get("ms_before", 0.0))
    ms_after = float(record.get("ms_after", num_samples))
    if ms_before + ms_after <= 0:
        return np.arange(num_samples, dtype=float), "Sample"
    return (
        np.linspace(-ms_before, ms_after, num_samples, endpoint=False),
        "Time (ms)",
    )


def integral_average(waveform, time_values):
    """Return integral(waveform) divided by the sampled time interval."""
    if waveform.size == 0:
        raise ValueError("Cannot average an empty waveform")
    if waveform.size < 2:
        return float(waveform[0])

    duration = float(time_values[-1] - time_values[0])
    if duration == 0.0:
        return float(np.mean(waveform))
    interval_integrals = (
        0.5 * (waveform[:-1] + waveform[1:]) * np.diff(time_values)
    )
    return float(np.sum(interval_integrals) / duration)


def draw_waveform(axis, record, title):
    waveform = np.asarray(record["raw_mean_template"], dtype=float)
    if waveform.ndim != 1:
        raise ValueError(
            "%s waveform must be one-dimensional, got shape %s"
            % (title, waveform.shape)
        )

    time_values, x_label = waveform_time_ms(record, waveform.size)
    average_level = integral_average(waveform, time_values)
    axis.plot(time_values, waveform, color="tab:blue", linewidth=1.7)
    axis.axhline(
        average_level,
        color="0.2",
        linestyle="--",
        linewidth=1.0,
    )
    axis.fill_between(
        time_values,
        waveform,
        average_level,
        where=waveform >= average_level,
        interpolate=True,
        color="tab:orange",
        alpha=0.3,
    )
    axis.fill_between(
        time_values,
        waveform,
        average_level,
        where=waveform < average_level,
        interpolate=True,
        color="tab:blue",
        alpha=0.25,
    )
    axis.axvline(0.0, color="0.55", linestyle="--", linewidth=0.8)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Raw signal")
    axis.grid(alpha=0.25)


def plot_waveforms(templates, indices, out_path):
    _, units = selected_units_from_indices(templates, indices)
    records = [get_record(templates, unit_id) for unit_id in units]

    num_cols = 3
    num_rows = int(math.ceil(len(indices) / float(num_cols)))
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(15, 3.5 * num_rows),
        squeeze=False,
    )

    for axis, index, unit_id, record in zip(axes.flat, indices, units, records):
        channel_id = record["primary_channel"]
        draw_waveform(
            axis,
            record,
            "Index %d, unit %d, channel %s" % (index, unit_id, channel_id),
        )

    for axis in list(axes.flat)[len(indices):]:
        axis.set_visible(False)

    fig.tight_layout()
    os.makedirs(out_path, exist_ok=True)
    output_file = os.path.join(
        out_path, "wave_%d_%d.png" % (indices[0], indices[-1])
    )
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_file


def plot_multichannel_candidate_waveforms(
    templates, off_grid_unit_ids, grid_offsets, out_path, random_seed
):
    num_plots = min(9, len(off_grid_unit_ids))
    rng = np.random.default_rng(random_seed)
    selected_indices = rng.choice(
        len(off_grid_unit_ids), size=num_plots, replace=False
    )
    selected_units = [off_grid_unit_ids[index] for index in selected_indices]

    num_cols = 3
    num_rows = max(1, int(math.ceil(num_plots / float(num_cols))))
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(15, 3.5 * num_rows),
        squeeze=False,
    )

    if num_plots == 0:
        axes.flat[0].text(
            0.5,
            0.5,
            "No off-grid multi-channel candidates",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes.flat[0].transAxes,
        )
        axes.flat[0].set_axis_off()
    else:
        for axis, unit_id in zip(axes.flat, selected_units):
            record = get_record(templates, unit_id)
            channel_id = record["primary_channel"]
            distance = grid_offsets[unit_id]["distance"]
            draw_waveform(
                axis,
                record,
                "Unit %s, channel %s, grid offset %.2f"
                % (unit_id, channel_id, distance),
            )

    num_visible_axes = max(1, num_plots)
    for axis in list(axes.flat)[num_visible_axes:]:
        axis.set_visible(False)

    fig.suptitle("Random off-grid multi-channel candidates", fontsize=14)
    fig.tight_layout()
    os.makedirs(out_path, exist_ok=True)
    output_file = os.path.join(out_path, "multiChan.png")
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_file, selected_units


def plot_node_locations(
    locations,
    units,
    indices,
    out_path,
    other_units_by_unit,
    off_grid_unit_ids,
):
    selected_locations = []
    for unit_id in units:
        key = mea_match_key(unit_id)
        if key not in locations:
            raise KeyError(
                "Selected unit %r was not found in metrics_curated.xlsx" % unit_id
            )
        selected_locations.append(locations[key])

    all_xy = np.asarray(list(locations.values()), dtype=float)
    selected_xy = np.asarray(selected_locations, dtype=float)

    fig, axis = plt.subplots(figsize=(10, 6))
    axis.scatter(
        all_xy[:, 0],
        all_xy[:, 1],
        s=35,
        facecolors="none",
        edgecolors="0.45",
        linewidths=0.9,
        label="All units",
    )
    if off_grid_unit_ids:
        off_grid_xy = np.asarray(
            [locations[unit_id] for unit_id in off_grid_unit_ids], dtype=float
        )
        axis.scatter(
            off_grid_xy[:, 0],
            off_grid_xy[:, 1],
            s=55,
            marker="x",
            color="tab:red",
            linewidths=1.5,
            zorder=2,
            label="Off-grid candidate",
        )
    axis.scatter(
        selected_xy[:, 0],
        selected_xy[:, 1],
        s=65,
        facecolors="none",
        edgecolors="tab:blue",
        linewidths=1.8,
        zorder=3,
        label="Selected unit",
    )
    for unit_id, (loc_x, loc_y) in zip(units, selected_locations):
        axis.annotate(
            "%s\n(+%d)" % (unit_id, other_units_by_unit[unit_id]),
            (loc_x, loc_y),
            xytext=(5, 0),
            textcoords="offset points",
            color="tab:blue",
            fontsize=9,
            fontweight="bold",
            verticalalignment="center",
            zorder=4,
        )

    axis.set_title(
        "Unit positions: %d total; %d off-grid candidates"
        % (len(locations), len(off_grid_unit_ids))
    )
    axis.set_xlabel("Unit x position (loc_x)")
    axis.set_ylabel("Unit y position (loc_y)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    fig.tight_layout()

    os.makedirs(out_path, exist_ok=True)
    output_file = os.path.join(
        out_path, "xyloc_%d_%d.png" % (indices[0], indices[-1])
    )
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_file


def main():
    args = parse_args()
    print("Data path:", os.path.abspath(args.rawPath))
    templates, _ = load_templates(args.rawPath)
    available_units, selected_units = selected_units_from_indices(
        templates, args.index
    )
    locations, _ = load_node_locations(args.rawPath, args.metricsPath)
    if not np.isfinite(args.gridTolerance) or args.gridTolerance < 0.0:
        raise ValueError("grid tolerance must be a finite non-negative number")
    grid_offsets = distances_from_electrode_grid(
        locations, args.gridPitch, args.gridOrigin
    )
    off_grid_unit_ids = [
        unit_id
        for unit_id, result in grid_offsets.items()
        if result["distance"] > args.gridTolerance
    ]
    channel_multiplicities, other_units_by_unit = channel_unit_multiplicities(
        templates, available_units
    )
    off_grid_unit_id_set = set(off_grid_unit_ids)
    units_not_sharing_channel = sum(
        other_units_by_unit[unit_id] == 0 for unit_id in available_units
    )
    units_not_sharing_or_multichannel = sum(
        other_units_by_unit[unit_id] == 0
        and mea_match_key(unit_id) not in off_grid_unit_id_set
        for unit_id in available_units
    )
    print("unit_mult  num_chan")
    for num_units in sorted(channel_multiplicities):
        print("%d %d" % (num_units, channel_multiplicities[num_units]))
    print("Total units:", len(available_units))
    print(
        "Grid pitch: x=%.6g, y=%.6g" % (args.gridPitch, args.gridPitch)
    )
    print(
        "Multi-channel classifier: distance from nearest grid point > %.6g"
        % args.gridTolerance
    )
    print("Summary:")
    print("Multi-channel units found:", len(off_grid_unit_ids))
    print("Multi-channel unit IDs:", " ".join(off_grid_unit_ids))
    print("Units not sharing their primary channel:", units_not_sharing_channel)
    print(
        "Units not sharing their primary channel and not multi-channel candidates:",
        units_not_sharing_or_multichannel,
    )
    waveform_file = plot_waveforms(templates, args.index, args.outPath)
    print("Saved PNG:", waveform_file)
    locations_file = plot_node_locations(
        locations,
        selected_units,
        args.index,
        args.outPath,
        other_units_by_unit,
        off_grid_unit_ids,
    )
    print("Saved PNG:", locations_file)
    multichannel_file, plotted_multichannel_units = (
        plot_multichannel_candidate_waveforms(
            templates,
            off_grid_unit_ids,
            grid_offsets,
            args.outPath,
            args.multiChanSeed,
        )
    )
    print(
        "Saved PNG: %s (units: %s)"
        % (multichannel_file, " ".join(plotted_multichannel_units))
    )


if __name__ == "__main__":
    main()
