import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_future_predictions_with_history(
    df: pd.DataFrame,                    # FUTURE data (your df_now), can include multiple games/cohorts
    scored_by_sd: dict,                  # from train_eval_models_by_snapshot_day: scored_by_sd[snapshot_day] -> df_history
    snapshot_day: int,
    thresholds_by_sd: dict,              # snapshot_day -> frozen threshold
    *,
    show_history: bool = True,
    show_actual_future: bool = True,    # if True and actual labels exist, future fill becomes red/green with BLUE outline
    agg_by_game: bool = True,            # True: one dot per game; False: one dot per row/cohort (jittered)
    x_offset: float = 0.18,              # offset for future dots when show_history=True
    jitter: float = 0.08,                # only used when agg_by_game=False
    random_state: int = 0,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None,
    return_fig: bool = True,
):
    """
    Visualize FUTURE predictions (blue) optionally overlaid on HISTORICAL evaluation (red/green).
    - Historical source: scored_by_sd[snapshot_day] with columns: game_id, p_breach, y_true (or roas_breach)
    - Future df must have: game_id, p_breach (or p_breach_raw). Optional: y_true/roas_breach for show_actual_future.

    Returns: (fig, ax) if return_fig else None
    """

    # ---------- helpers ----------
    def _get_label_col(d: pd.DataFrame) -> str | None:
        for c in ["y_true", "roas_breach"]:
            if c in d.columns:
                return c
        return None

    def _require_cols(d: pd.DataFrame, cols: list[str], name: str):
        missing = [c for c in cols if c not in d.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")

    # ---------- threshold ----------
    if snapshot_day not in thresholds_by_sd:
        raise ValueError(f"thresholds_by_sd has no entry for snapshot_day={snapshot_day}")
    thr = float(thresholds_by_sd[snapshot_day])

    # ---------- future df ----------
    df_fut = df.copy()
    if "snapshot_day" in df_fut.columns:
        df_fut = df_fut[df_fut["snapshot_day"] == snapshot_day].copy()
    if df_fut.empty:
        raise ValueError(f"Future df is empty for snapshot_day={snapshot_day}")

    # choose probability column
    p_col = "p_breach" if "p_breach" in df_fut.columns else ("p_breach_raw" if "p_breach_raw" in df_fut.columns else None)
    if p_col is None:
        raise ValueError("Future df must contain 'p_breach' (recommended) or 'p_breach_raw'")
    _require_cols(df_fut, ["game_id", p_col], "Future df")

    fut_label_col = _get_label_col(df_fut)
    hist_label_col = None

    # ---------- history df ----------
    df_hist = None
    if show_history:
        if snapshot_day not in scored_by_sd:
            raise ValueError(f"scored_by_sd has no entry for snapshot_day={snapshot_day}")
        df_hist = scored_by_sd[snapshot_day].copy()
        # history always uses p_breach
        _require_cols(df_hist, ["game_id", "p_breach"], "History df")
        hist_label_col = _get_label_col(df_hist)
        if hist_label_col is None:
            raise ValueError("History df must contain 'y_true' or 'roas_breach'")

    # ---------- determine game order ----------
    if show_history:
      game_order = sorted(df_hist["game_id"].astype(str).unique().tolist())
    else:
      game_order = sorted(df_fut["game_id"].astype(str).unique().tolist())


    # absolute value ordering
    '''
    if show_history:
        order_src = (
            df_hist.groupby("game_id", as_index=True)["p_breach"]
            .mean()
            .sort_values(ascending=False)
        )
    else:
        order_src = (
            df_fut.groupby("game_id", as_index=True)[p_col]
            .mean()
            .sort_values(ascending=False)
        )

    game_order = order_src.index.tolist()
    '''
    # ---------- plotting ----------
    if figsize is None:
        n_games = len(game_order)
        figsize = (max(10, n_games * 0.25), 5)

    fig, ax = plt.subplots(figsize=figsize)

    # ===== agg_by_game =====
    if agg_by_game:
        x = np.arange(len(game_order))

        # --- history layer ---
        if show_history:
            g_hist = df_hist.groupby("game_id", as_index=False).agg(
                p_mean=("p_breach", "mean"),
                y_rate=(hist_label_col, "mean"),
                n=(hist_label_col, "size"),
            )
            g_hist["y_major"] = (g_hist["y_rate"] >= 0.5).astype(int)
            g_hist["game_id"] = pd.Categorical(g_hist["game_id"], categories=game_order, ordered=True)
            g_hist = g_hist.sort_values("game_id")

            x_hist = np.arange(len(g_hist))
            y_hist = g_hist["p_mean"].to_numpy()
            c_hist = np.where(g_hist["y_major"].to_numpy() == 1, "red", "green")
            ax.scatter(x_hist, y_hist, c=c_hist, alpha=0.85, label="historic (actual)")

        # --- future layer ---
        g_fut = df_fut.groupby("game_id", as_index=False).agg(
            p_mean=(p_col, "mean"),
            n=(p_col, "size"),
            **({ "y_rate": (fut_label_col, "mean") } if (show_actual_future and fut_label_col is not None) else {})
        )
        if show_actual_future and fut_label_col is not None:
            g_fut["y_major"] = (g_fut["y_rate"] >= 0.5).astype(int)

        g_fut["game_id"] = pd.Categorical(g_fut["game_id"], categories=game_order, ordered=True)
        g_fut = g_fut.sort_values("game_id")

        x_fut = np.arange(len(g_fut)) + (x_offset if show_history else 0.0)
        y_fut = g_fut["p_mean"].to_numpy()

        if show_actual_future and fut_label_col is not None:
            face = np.where(g_fut["y_major"].to_numpy() == 1, "red", "green")
            ax.scatter(x_fut, y_fut, c=face, edgecolors="blue", linewidths=1.2, alpha=0.9, label="pred (actual w/ blue edge)")
        else:
            ax.scatter(x_fut, y_fut, c="blue", alpha=0.9, label="pred")

        ax.set_xticks(np.arange(len(game_order)))
        ax.set_xticklabels([str(g) for g in game_order], rotation=90, ha="center")
        ax.set_xlabel("game_id (sorted by mean p)")

    # ===== row-level (jittered) =====
    else:
        rng = np.random.RandomState(random_state)

        # --- history ---
        if show_history:
            dfh = df_hist.copy()
            dfh["game_id"] = pd.Categorical(dfh["game_id"], categories=game_order, ordered=True)
            dfh = dfh.sort_values("game_id")
            x0 = dfh["game_id"].cat.codes.to_numpy().astype(float)
            xh = x0 + (rng.randn(len(x0)) * jitter)
            yh = dfh["p_breach"].to_numpy()
            ch = np.where(dfh[hist_label_col].to_numpy().astype(int) == 1, "red", "green")
            ax.scatter(xh, yh, c=ch, alpha=0.6, s=18, label="historic (actual)")

        # --- future ---
        dff = df_fut.copy()
        dff["game_id"] = pd.Categorical(dff["game_id"], categories=game_order, ordered=True)
        dff = dff.sort_values("game_id")
        x0 = dff["game_id"].cat.codes.to_numpy().astype(float)
        xf = x0 + (rng.randn(len(x0)) * jitter) + (x_offset if show_history else 0.0)
        yf = dff[p_col].to_numpy()

        if show_actual_future and fut_label_col is not None:
            ytrue = dff[fut_label_col].to_numpy().astype(int)
            face = np.where(ytrue == 1, "red", "green")
            ax.scatter(xf, yf, c=face, edgecolors="blue", linewidths=1.0, alpha=0.85, s=28, label="current (actual w/ blue edge)")
        else:
            ax.scatter(xf, yf, c="blue", alpha=0.85, s=28, label="current")

        ax.set_xticks(np.arange(len(game_order)))
        ax.set_xticklabels([str(g) for g in game_order], rotation=90, ha="center")
        ax.set_xlabel("game_id (sorted by mean p)")

    # common styling
    ax.axhline(thr, linestyle="--")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Predicted breach probability")
    ax.set_title(title or f"Snapshot day {snapshot_day}: historic vs current pred (thr={thr:.3f})")

    # keep legend compact
    ax.legend(loc="lower left", fontsize=9, frameon=False)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=160)

    if return_fig:
        return fig, ax
    plt.close(fig)
    return None