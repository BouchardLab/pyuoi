def make_canvas1(fitD, fitMD, spikes, S_true, C_true, eta_clip, T_show):
    """
    Layout (4 rows):
      Row 0 (tall) : observed spike raster  - all neurons
      Row 1 (tall) : predicted rate heatmap | eta heatmap  - all neurons
      Row 2 (med)  : C_true mixing coefficients
      Row 3 (thin) : true state trace
    All time axes show T_show bins.
    """
    lambda_t = fitD["lambda_t"]   # (T, N)
    eta_t    = fitD["eta_t"]      # (T, N)

    T, N    = spikes.shape
    T_show  = min(T_show, T)
    M       = C_true.shape[1]

    fig = plt.figure(figsize=(16, 13))
    fig.suptitle(make_title("Time-Domain Panels", fitMD, eta_clip),
                 fontsize=10, y=1.00)

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           height_ratios=[2.8, 2.8, 1.4, 0.55],
                           hspace=0.38, wspace=0.28)

    # ---- Row 0: observed raster (spans both columns, all neurons) ----
    ax_obs = fig.add_subplot(gs[0, :])
    t_idx, n_idx = np.where(spikes[:T_show, :N] > 0)
    ax_obs.scatter(t_idx, n_idx, s=0.8, color="black", alpha=0.45)
    ax_obs.set_xlim(0, T_show)
    ax_obs.set_ylim(-0.5, N - 0.5)
    ax_obs.set_ylabel("Neuron index")
    ax_obs.set_title(f"Observed spikes  (all {N} neurons)")
    ax_obs.set_xticklabels([])
    # mark E/I boundary if detectable - simple heuristic: 
    # high-rate neurons tend to be excitatory
    mean_rate = spikes[:T_show, :].mean(axis=0)
    boundary  = int(np.sum(mean_rate > mean_rate.mean()))
    ax_obs.axhline(boundary, color="red", lw=0.8, ls="--", alpha=0.6,
                   label=f"rate boundary ~{boundary}")
    ax_obs.legend(fontsize=7, loc="upper right")

    # ---- Row 1 left: predicted rate heatmap - all neurons ----
    ax_lam = fig.add_subplot(gs[1, 0])
    im1 = ax_lam.imshow(lambda_t[:T_show, :N].T,
                        aspect="auto", origin="lower",
                        extent=[0, T_show, 0, N],
                        cmap="hot", interpolation="nearest")
    plt.colorbar(im1, ax=ax_lam, label="λ (sp/bin)", pad=0.02)
    ax_lam.set_ylabel("Neuron index")
    ax_lam.set_title(f"Predicted rate  λ_t  (all {N} neurons)")
    ax_lam.set_xticklabels([])

    # ---- Row 1 right: eta heatmap - all neurons ----
    ax_eta = fig.add_subplot(gs[1, 1])
    im2 = ax_eta.imshow(eta_t[:T_show, :N].T,
                        aspect="auto", origin="lower",
                        extent=[0, T_show, 0, N],
                        cmap="RdBu_r", interpolation="nearest",
                        vmin=-eta_clip, vmax=eta_clip)
    plt.colorbar(im2, ax=ax_eta, label="η", pad=0.02)
    ax_eta.set_ylabel("Neuron index")
    ax_eta.set_title(f"Internal potential  η_t  (all {N} neurons)")
    ax_eta.set_xticklabels([])

    # ---- Row 2: mixing coefficients (spans both columns) ----
    ax_c = fig.add_subplot(gs[2, :])
    colors = plt.cm.tab10(np.linspace(0, 0.5, M))
    for m in range(M):
        ax_c.plot(C_true[:T_show, m], lw=1.1,
                  color=colors[m], label=f"c_{m}", alpha=0.85)
    ax_c.set_xlim(0, T_show)
    ax_c.set_ylim(-0.05, 1.05)
    ax_c.set_ylabel("Coefficient value")
    ax_c.set_title("C_true  mixing coefficients")
    ax_c.legend(fontsize=8, loc="upper right", ncol=M)
    ax_c.set_xticklabels([])

    # ---- Row 3: true state trace (spans both columns) ----
    ax_s = fig.add_subplot(gs[3, :])
    ax_s.step(np.arange(T_show), S_true[:T_show],
              where="mid", color="steelblue", lw=1)
    ax_s.set_xlim(0, T_show)
    ax_s.set_ylabel("State", fontsize=8)
    ax_s.set_xlabel(f"Time step  (first {T_show} of {T})", fontsize=9)
    ax_s.yaxis.set_major_locator(MaxNLocator(integer=True))

    return fig
