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
        help="Plot letters; a=A off-diagonal/diagonal and B state comparisons",
    )
    return parser.parse_args()


def normalize_plot_letters(show_plots):
    letters = "".join(str(item) for item in show_plots).replace(" ", "")
    unknown = sorted(set(letters) - {"a"})
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
    return A, B, inp_f


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


def correlation_panel(ax, x, y, title, x_label, y_label, color):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    assert x.shape == y.shape
    finite = np.isfinite(x) & np.isfinite(y)
    xf = x[finite]
    yf = y[finite]
    r_value, n_value = correlation_safe(x, y)

    ax.scatter(xf, yf, s=8, alpha=0.40, color=color, edgecolors="none")
    if n_value:
        # Preserve independent x/y autoscaling while showing y=x where the
        # two visible numeric ranges overlap.
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
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
    ax.text(
        0.04, 0.96, "Pearson r=%s\nn=%d" % ("%.4f" % r_value, n_value),
        transform=ax.transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.8),
    )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)


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
    A1, B1, inp_f1 = load_debias_fit(args.basePath, args.dataName1)
    A2, B2, inp_f2 = load_debias_fit(args.basePath, args.dataName2)
    assert_matching_dimensions(A1, B1, A2, B2)

    print("Loaded fit 1: %s  A=%s B=%s" % (inp_f1, A1.shape, B1.shape))
    print("Loaded fit 2: %s  A=%s B=%s" % (inp_f2, A2.shape, B2.shape))

    if "a" in plot_letters:
        import matplotlib

        if not os.environ.get("DISPLAY"):
            matplotlib.use("Agg")
        fig = plot_a(A1, B1, A2, B2, args.dataName1, args.dataName2)
        out_dir = os.path.join(args.basePath, "plots")
        os.makedirs(out_dir, exist_ok=True)
        out_f = output_file(args.basePath, out_stem, "a")
        fig.savefig(out_f, dpi=150)
        print("Saved comparison canvas: %s" % out_f)

        if os.environ.get("DISPLAY") and "agg" not in matplotlib.get_backend().lower():
            import matplotlib.pyplot as plt
            plt.show()


if __name__ == "__main__":
    main()
