#!/usr/bin/env python3
"""Compare the fitted A and B values from two Stage (c) de-biased fits."""

import argparse
import os
import re
import secrets

import numpy as np

from toolbox.Util_NumpyIOv2 import read_data_npz


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two prism_deBiasFit3c.py fit results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--basePath", required=True,
        help="Run directory containing prismFit/ and plots/",
    )
    parser.add_argument(
        "--dataName1", required=True,
        help="First Stage (c) fit stem in prismFit/, used on the x axis",
    )
    parser.add_argument(
        "--dataName2", required=True,
        help="Second Stage (c) fit stem in prismFit/, used on the y axis",
    )
    parser.add_argument(
        "--outName", default=None,
        help="Output PNG stem in plots/; default: compare_<random hash6>",
    )
    parser.add_argument(
        "-p", "--showPlots", nargs="+", default=["a"],
        help=(
            "Plot letters; a=A off-diagonal/diagonal and B state comparisons; "
            "b=A off-diagonal comparisons for data1 excitatory/inhibitory source columns"
        ),
    )
    return parser.parse_args()


def normalize_plot_letters(show_plots):
    letters = "".join(str(item) for item in show_plots).replace(" ", "")
    unknown = sorted(set(letters) - {"a", "b"})
    assert not unknown, "Unknown plot letters: %s" % "".join(unknown)
    return letters


def fit_file(base_path, data_name):
    return os.path.join(base_path, "prismFit", "%s.prismEM.npz" % data_name)


def load_debias_fit(base_path, data_name):
    inp_f = fit_file(base_path, data_name)
    fit_d, fit_md = read_data_npz(inp_f, verb=False)
    assert "A_debias" in fit_d, "%s must contain A_debias" % inp_f
    assert "B_debias" in fit_d, "%s must contain B_debias" % inp_f
    if isinstance(fit_md, dict):
        assert (
            fit_md.get("fit_type") == "prismEM_deBias_stageC"
            or "deBias_stageC" in fit_md
        ), "%s is not a prism_deBiasFit3c.py Stage (c) result" % inp_f

    A = np.asarray(fit_d["A_debias"], dtype=np.float64)
    B = np.asarray(fit_d["B_debias"], dtype=np.float64)
    if B.ndim == 1:
        B = B[None, :]

    assert A.ndim == 2 and A.shape[0] == A.shape[1], (
        "%s A_debias must be square 2D, got %s" % (inp_f, A.shape)
    )
    assert B.ndim == 2 and B.shape[0] >= 1, (
        "%s B_debias must contain at least one state, got %s" % (inp_f, B.shape)
    )
    assert B.shape[1] == A.shape[0], (
        "%s B_debias neuron dimension %d does not match A_debias dimension %d"
        % (inp_f, B.shape[1], A.shape[0])
    )
    neuron_type = fit_d.get("neuron_type")
    if neuron_type is not None:
        neuron_type = np.asarray(neuron_type, dtype=np.int8).reshape(-1)
    return A, B, neuron_type, fit_md, inp_f


def load_neuron_unit_ids(base_path, fit_md, num_neurons, fit_file_name):
    assert isinstance(fit_md, dict), "%s metadata must be a dictionary" % fit_file_name
    assert fit_md.get("data_type") == "bioExp", (
        "%s must be a biological fit with data_type='bioExp' to verify unit IDs"
        % fit_file_name
    )
    provenance = fit_md.get("provenance", {})
    experiment_name = provenance.get("experiment_name")
    assert experiment_name, "%s metadata is missing provenance.experiment_name" % fit_file_name

    node_f = os.path.join(base_path, "spikesData", "%s.bioExp.npz" % experiment_name)
    node_d, _ = read_data_npz(node_f, verb=False)
    assert "MEA_idx" in node_d, "%s must contain MEA_idx unit IDs" % node_f
    unit_ids = np.asarray(node_d["MEA_idx"]).reshape(-1)
    assert unit_ids.size == num_neurons, (
        "%s MEA_idx has %d unit IDs but %s has %d neuron columns"
        % (node_f, unit_ids.size, fit_file_name, num_neurons)
    )
    return unit_ids


def count_unit_id_mismatches(unit_ids1, unit_ids2):
    unit_ids1 = np.asarray(unit_ids1).reshape(-1)
    unit_ids2 = np.asarray(unit_ids2).reshape(-1)
    overlap = min(unit_ids1.size, unit_ids2.size)
    mismatch_count = abs(unit_ids1.size - unit_ids2.size)
    mismatch_count += sum(
        not np.array_equal(unit_ids1[index], unit_ids2[index])
        for index in range(overlap)
    )
    return int(mismatch_count), int(max(unit_ids1.size, unit_ids2.size))


def validate_neuron_type(neuron_type, num_neurons, fit_file_name):
    assert neuron_type is not None, "%s must contain neuron_type for plot b" % fit_file_name
    neuron_type = np.asarray(neuron_type, dtype=np.int8).reshape(-1)
    assert neuron_type.shape == (num_neurons,), (
        "%s neuron_type shape %s does not match %d A-matrix columns"
        % (fit_file_name, neuron_type.shape, num_neurons)
    )
    return neuron_type


def jaccard_index(mask1, mask2):
    mask1 = np.asarray(mask1, dtype=bool).reshape(-1)
    mask2 = np.asarray(mask2, dtype=bool).reshape(-1)
    assert mask1.shape == mask2.shape
    union_count = int(np.count_nonzero(mask1 | mask2))
    if union_count == 0:
        return np.nan
    return float(np.count_nonzero(mask1 & mask2)) / union_count


def source_type_offdiagonal_values(A1, A2, neuron_type1, type_value):
    assert A1.shape == A2.shape
    assert A1.ndim == 2 and A1.shape[0] == A1.shape[1]
    neuron_type1 = np.asarray(neuron_type1, dtype=np.int8).reshape(-1)
    assert neuron_type1.shape == (A1.shape[1],)
    assert type_value in (-1, 1)
    off_diagonal = ~np.eye(A1.shape[0], dtype=bool)
    data1_type_columns = neuron_type1 == type_value
    selected = off_diagonal & data1_type_columns[None, :]
    values1 = A1[selected]
    values2 = A2[selected]
    both_nonzero = (values1 != 0.0) & (values2 != 0.0)
    return values1[both_nonzero], values2[both_nonzero]


def excitatory_offdiagonal_values(A1, A2, neuron_type1):
    return source_type_offdiagonal_values(A1, A2, neuron_type1, 1)


def inhibitory_offdiagonal_values(A1, A2, neuron_type1):
    return source_type_offdiagonal_values(A1, A2, neuron_type1, -1)


def assert_matching_dimensions(A1, B1, A2, B2):
    assert A1.shape == A2.shape, (
        "A_debias dimensions differ: fit 1 %s vs fit 2 %s" % (A1.shape, A2.shape)
    )
    assert B1.shape == B2.shape, (
        "B_debias dimensions differ: fit 1 %s vs fit 2 %s" % (B1.shape, B2.shape)
    )


def correlation_safe(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    assert x.shape == y.shape
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan, int(x.size)
    return float(np.corrcoef(x, y)[0, 1]), int(x.size)


def correlation_panel(
    ax, x, y, title, x_label, y_label, color, extra_stats=None,
    extend_y_from_x=None,
):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    assert x.shape == y.shape
    finite = np.isfinite(x) & np.isfinite(y)
    xf = x[finite]
    yf = y[finite]
    r_value, n_value = correlation_safe(x, y)

    ax.scatter(xf, yf, s=8, alpha=0.40, color=color, edgecolors="none")
    if n_value:
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        if extend_y_from_x is not None:
            assert extend_y_from_x in ("positive", "negative")
            extra_range = 0.2 * (x_lim[1] - x_lim[0])
            if extend_y_from_x == "positive":
                y_lim = (x_lim[0], x_lim[1] + extra_range)
            else:
                y_lim = (x_lim[0] - extra_range, x_lim[1])
            line_lo, line_hi = x_lim
        else:
            # Show y=x where the independently autoscaled ranges overlap.
            line_lo = max(x_lim[0], y_lim[0])
            line_hi = min(x_lim[1], y_lim[1])
        if line_lo < line_hi:
            ax.plot(
                [line_lo, line_hi], [line_lo, line_hi], "--",
                color="k", linewidth=0.9, alpha=0.7,
            )
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)

    ax.axhline(0.0, color="k", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.axvline(0.0, color="k", linestyle=":", linewidth=0.7, alpha=0.5)
    stats_text = "Pearson r=%s\nn=%d" % ("%.4f" % r_value, n_value)
    if extra_stats:
        stats_text += "\n%s" % extra_stats
    ax.text(
        0.04, 0.96, stats_text,
        transform=ax.transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.8),
    )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    # Use the same physical scale for x and y so the fit comparison and the
    # y=x reference are not distorted by a rectangular data aspect.
    ax.set_aspect("equal", adjustable="box")


def plot_a(A1, B1, A2, B2, data_name1, data_name2):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2, 2, figsize=(11, 9), facecolor="white", constrained_layout=True,
    )
    diag_mask = np.eye(A1.shape[0], dtype=bool)
    x_label = "data1"
    y_label = "data2"

    correlation_panel(
        axes[0, 0], A1[~diag_mask], A2[~diag_mask],
        "A off-diagonal elements", x_label, y_label, "tab:green",
    )
    correlation_panel(
        axes[0, 1], A1[diag_mask], A2[diag_mask],
        "A diagonal elements", x_label, y_label, "tab:purple",
    )
    correlation_panel(
        axes[1, 0], B1[0], B2[0],
        "B elements, state 0", x_label, y_label, "tab:blue",
    )

    if B1.shape[0] >= 2:
        correlation_panel(
            axes[1, 1], B1[1], B2[1],
            "B elements, state 1", x_label, y_label, "tab:orange",
        )
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.5, 0.5, "B state 1 is not present\n(num_states=1)",
            transform=axes[1, 1].transAxes, ha="center", va="center", fontsize=12,
        )

    fig.suptitle(
        "Stage (c) fit comparison\ndata1: %s\ndata2: %s" % (data_name1, data_name2),
        fontsize=13,
    )
    return fig


def plot_b(A1, A2, neuron_type1, neuron_type2, data_name1, data_name2):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 5), facecolor="white", constrained_layout=True,
    )
    exc1 = np.asarray(neuron_type1).reshape(-1) > 0
    exc2 = np.asarray(neuron_type2).reshape(-1) > 0
    inh1 = np.asarray(neuron_type1).reshape(-1) < 0
    inh2 = np.asarray(neuron_type2).reshape(-1) < 0
    exc_jaccard = jaccard_index(exc1, exc2)
    inh_jaccard = jaccard_index(inh1, inh2)
    exc_values1, exc_values2 = excitatory_offdiagonal_values(A1, A2, neuron_type1)
    inh_values1, inh_values2 = inhibitory_offdiagonal_values(A1, A2, neuron_type1)
    correlation_panel(
        axes[0], exc_values1, exc_values2,
        "A off-diagonal: data1 excitatory sources, both nonzero",
        "data1", "data2", "tab:red",
        extra_stats=(
            "Exc Jaccard=%s\nexc: data1=%d, data2=%d"
            % ("%.4f" % exc_jaccard, np.count_nonzero(exc1), np.count_nonzero(exc2))
        ),
        extend_y_from_x="positive",
    )
    correlation_panel(
        axes[1], inh_values1, inh_values2,
        "A off-diagonal: data1 inhibitory sources, both nonzero",
        "data1", "data2", "tab:blue",
        extra_stats=(
            "Inh Jaccard=%s\ninh: data1=%d, data2=%d"
            % ("%.4f" % inh_jaccard, np.count_nonzero(inh1), np.count_nonzero(inh2))
        ),
        extend_y_from_x="negative",
    )

    fig.suptitle(
        "Stage (c) excitatory/inhibitory-source comparison\ndata1: %s\ndata2: %s"
        % (data_name1, data_name2),
        fontsize=13,
    )
    return fig


def resolve_output_stem(out_name):
    if out_name is None:
        return "compare_%s" % secrets.token_hex(3)
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", out_name), (
        "outName must be a file stem containing only letters, digits, underscores, dots, or hyphens"
    )
    return out_name


def output_file(base_path, out_stem, plot_letter):
    return os.path.join(base_path, "plots", "%s_%s.png" % (out_stem, plot_letter))


def main():
    args = parse_args()
    plot_letters = normalize_plot_letters(args.showPlots)
    out_stem = resolve_output_stem(args.outName)
    assert args.dataName1 != args.dataName2, "dataName1 and dataName2 must identify different fits"
    A1, B1, neuron_type1, fit_md1, inp_f1 = load_debias_fit(
        args.basePath, args.dataName1
    )
    A2, B2, neuron_type2, fit_md2, inp_f2 = load_debias_fit(
        args.basePath, args.dataName2
    )
    unit_ids1 = load_neuron_unit_ids(args.basePath, fit_md1, A1.shape[1], inp_f1)
    unit_ids2 = load_neuron_unit_ids(args.basePath, fit_md2, A2.shape[1], inp_f2)
    num_mismatches, num_positions = count_unit_id_mismatches(unit_ids1, unit_ids2)
    if num_mismatches:
        print(
            "Neuron column/unit ID mismatch: %d of %d positions differ; exiting without plots."
            % (num_mismatches, num_positions)
        )
        return
    print("Neuron column/unit IDs match at all %d positions." % num_positions)
    assert_matching_dimensions(A1, B1, A2, B2)

    print("Loaded fit 1: %s  A=%s B=%s" % (inp_f1, A1.shape, B1.shape))
    print("Loaded fit 2: %s  A=%s B=%s" % (inp_f2, A2.shape, B2.shape))

    if plot_letters:
        import matplotlib

        if not os.environ.get("DISPLAY"):
            matplotlib.use("Agg")

    out_dir = os.path.join(args.basePath, "plots")

    if "a" in plot_letters:
        fig = plot_a(A1, B1, A2, B2, args.dataName1, args.dataName2)
        os.makedirs(out_dir, exist_ok=True)
        out_f = output_file(args.basePath, out_stem, "a")
        fig.savefig(out_f, dpi=150)
        print("Saved comparison canvas: %s" % out_f)

    if "b" in plot_letters:
        neuron_type1 = validate_neuron_type(neuron_type1, A1.shape[1], inp_f1)
        neuron_type2 = validate_neuron_type(neuron_type2, A2.shape[1], inp_f2)
        fig = plot_b(
            A1, A2, neuron_type1, neuron_type2, args.dataName1, args.dataName2
        )
        os.makedirs(out_dir, exist_ok=True)
        out_f = output_file(args.basePath, out_stem, "b")
        fig.savefig(out_f, dpi=150)
        print("Saved comparison canvas: %s" % out_f)

    if (
        os.environ.get("DISPLAY")
        and "agg" not in matplotlib.get_backend().lower()
    ):
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
