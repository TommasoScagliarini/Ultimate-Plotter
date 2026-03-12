"""
plotters.py
===========
Funzioni plotter per l'analisi della protesi SEA.

Ogni funzione crea, mostra (plt.show()) e salva una figura.

Grafici disponibili
-------------------
1. plot_sea_torque_time()    — 4×2 — coppia SEA, control input, reserve, sovrapposizione  [vs tempo]
2. plot_motor_kinematics()   — 5×2 — angoli/velocità joint e motore                       [vs tempo]
3. plot_torque_angle_power() — 2×2 — coppia su angolo e power su GC                       [su GC]
4. plot_kinematics_power()   — 3×2 — angolo, velocità, power                              [su GC]
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from utils import (
    add_gc_events, style_ax, add_legend,
    compute_power, cycle_stats, get_all_cycles, detect_IC,
)

# ── Colori fissi ─────────────────────────────────────────────────────────────
C = {
    "ankle_sea":  "#1a6fbd",
    "knee_sea":   "#d94f3d",
    "ankle_san":  "#f5a623",
    "knee_san":   "#2ca02c",
    "control":    "#26cc2b",
    "reserve":    "#8c564b",
    "overlay":    "#17becf",
    "gen":        "#2ca02c",
    "abs":        "#d94f3d",
    "motor":      "#0efbff",
    "zero":       "#aaaaaa",
}

# ── Helpers interni ───────────────────────────────────────────────────────────

def _save_and_show(fig: plt.Figure, out_dir: str, filename: str) -> None:
    """Mostra la figura e la salva in out_dir/filename."""
    plt.tight_layout()
    plt.show()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Salvato: {path}")


def _new_fig(nrows: int, ncols: int, figsize: tuple) -> tuple:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor("white")
    return fig, axes


def _plot_cycle_mean(ax, pct, mean, std, cycles, color,
                     plot_individuals: bool = True) -> None:
    """Traccia cicli individuali, banda ±1 SD e media."""
    if plot_individuals:
        for c in cycles:
            ax.plot(pct, c, color=color, lw=0.8, alpha=0.22)
    ax.fill_between(pct, mean - std, mean + std, color=color, alpha=0.18, label="±1 SD")
    ax.plot(pct, mean, color=color, lw=2.5, label="Media")


def _power_fill(ax, pct, mean) -> None:
    """Riempie le aree di generazione (verde) e assorbimento (rosso) sulla media."""
    ax.fill_between(pct, mean, 0, where=(mean >= 0),
                    color=C["gen"], alpha=0.15, label="Generazione")
    ax.fill_between(pct, mean, 0, where=(mean < 0),
                    color=C["abs"], alpha=0.15, label="Assorbimento")


def _col_labels(fig, titles: list[str], colors: list[str]) -> None:
    """Aggiunge titoli di colonna sopra la figura."""
    xs = [0.27, 0.73]
    for x, title, color in zip(xs, titles, colors):
        fig.text(x, 1.002, title, ha="center", fontsize=13,
                 fontweight="bold", color=color)


# ─────────────────────────────────────────────────────────────────────────────
# GRAFICO 1 — 4×2 — Coppia/Control/Reserve vs Tempo
# ─────────────────────────────────────────────────────────────────────────────

def plot_sea_torque_time(
    act_df,          # DataFrame Actuation_force
    ctrl_df,         # DataFrame Controls
    cfg: dict,
    out_dir: str,
) -> None:
    """
    Subplot 4×2 — tutto vs tempo di simulazione:
      Riga 1: SEA torque (SEA_Ankle | SEA_Knee)
      Riga 2: Control input (SEA_Ankle da controls | SEA_Knee da controls)
      Riga 3: Reserve actuator (reserve_pros_ankle | reserve_pros_knee)
      Riga 4: SEA torque + Reserve sovrapposti
    """
    print("[1/4] plot_sea_torque_time ...")

    t = act_df["time"].values

    ch = cfg["channels"]
    sea_a   = act_df[ch["sea_ankle"]].values
    sea_k   = act_df[ch["sea_knee"]].values
    ctrl_a  = ctrl_df[ch["ctrl_ankle"]].values
    ctrl_k  = ctrl_df[ch["ctrl_knee"]].values
    res_a   = act_df[ch["res_ankle"]].values
    res_k   = act_df[ch["res_knee"]].values

    fig, axes = _new_fig(4, 2, (20, 24))
    col_data = [
        # (ax,         signal,  color,        ylabel,             title)
        (axes[0][0], sea_a,  C["ankle_sea"], "Coppia (Nm)",      "SEA Torque — Ankle"),
        (axes[0][1], sea_k,  C["knee_sea"],  "Coppia (Nm)",      "SEA Torque — Knee"),
        (axes[1][0], ctrl_a, C["control"],   "Control input",    "Control Input — Ankle"),
        (axes[1][1], ctrl_k, C["control"],   "Control input",    "Control Input — Knee"),
        (axes[2][0], res_a,  C["reserve"],   "Coppia (Nm)",      "Reserve Actuator — Ankle"),
        (axes[2][1], res_k,  C["reserve"],   "Coppia (Nm)",      "Reserve Actuator — Knee"),
    ]

    for ax, sig, color, ylabel, title in col_data:
        ax.plot(t, sig, color=color, lw=1.5)
        style_ax(ax, "Tempo (s)", ylabel, title)

    # Riga 4: sovrapposizione SEA + Reserve
    for ax, sea, res, color_sea, label_sea in [
        (axes[3][0], sea_a, res_a, C["ankle_sea"], "SEA Ankle"),
        (axes[3][1], sea_k, res_k, C["knee_sea"],  "SEA Knee"),
    ]:
        ax.plot(t, sea, color=color_sea, lw=1.8, label=label_sea)
        ax.plot(t, res, color=C["reserve"],  lw=1.5, ls="--", alpha=0.8, label="Reserve")
        style_ax(ax, "Tempo (s)", "Coppia (Nm)", f"SEA + Reserve — {'Ankle' if 'Ankle' in label_sea else 'Knee'}")
        add_legend(ax)

    _col_labels(fig, ["Caviglia (Ankle)", "Ginocchio (Knee)"],
                [C["ankle_sea"], C["knee_sea"]])
    fig.suptitle("SEA Torque | Control Input | Reserve — vs Tempo",
                 color="black", fontsize=13, fontweight="bold", y=1.03)
    _save_and_show(fig, out_dir, "plot1_torque_time.png")


# ─────────────────────────────────────────────────────────────────────────────
# GRAFICO 2 — 5×2 — Cinematica joint e motore vs Tempo
# ─────────────────────────────────────────────────────────────────────────────

def plot_motor_kinematics(
    kin_q_df,        # DataFrame Kinematics_q
    kin_u_df,        # DataFrame Kinematics_u
    states_df,       # DataFrame States (può essere None)
    cfg: dict,
    out_dir: str,
) -> None:
    """
    Subplot 5×2 — tutto vs tempo di simulazione:
      Riga 1: Angolo joint
      Riga 2: Velocità joint
      Riga 3: Angolo motore SEA  (da states, skip se None)
      Riga 4: Velocità motore SEA (da states, skip se None)
      Riga 5: Angolo motore + Angolo joint sovrapposti
    """
    print("[2/4] plot_motor_kinematics ...")

    t_q = kin_q_df["time"].values
    t_u = kin_u_df["time"].values

    ch   = cfg["channels"]
    q_a  = -kin_q_df[ch["joint_ankle_q"]].values   # flessione +
    q_k  = -kin_q_df[ch["joint_knee_q"]].values
    u_a  = -kin_u_df[ch["joint_ankle_u"]].values
    u_k  = -kin_u_df[ch["joint_knee_u"]].values

    has_states = states_df is not None
    if has_states:
        t_s   = states_df["time"].values
        mq_a  = -states_df[ch["motor_angle_ankle"]].values
        mq_k  = -states_df[ch["motor_angle_knee"]].values
        mu_a  = -states_df[ch["motor_speed_ankle"]].values
        mu_k  = -states_df[ch["motor_speed_knee"]].values

    fig, axes = _new_fig(5, 2, (20, 30))

    # Riga 1 — Angolo joint
    for ax, sig, t, color, title in [
        (axes[0][0], q_a, t_q, C["ankle_sea"], "Angolo Joint — Ankle"),
        (axes[0][1], q_k, t_q, C["knee_sea"],  "Angolo Joint — Knee"),
    ]:
        ax.plot(t, sig, color=color, lw=1.5)
        style_ax(ax, "Tempo (s)", "Angolo (°)  [fless. +]", title)

    # Riga 2 — Velocità joint
    for ax, sig, t, color, title in [
        (axes[1][0], u_a, t_u, C["ankle_sea"], "Vel. Angolare Joint — Ankle"),
        (axes[1][1], u_k, t_u, C["knee_sea"],  "Vel. Angolare Joint — Knee"),
    ]:
        ax.plot(t, sig, color=color, lw=1.5)
        style_ax(ax, "Tempo (s)", "Vel. (°/s)  [fless. +]", title)

    # Riga 3 — Angolo motore
    for ax, col in [(axes[2][0], 0), (axes[2][1], 1)]:
        if has_states:
            sig   = mq_a if col == 0 else mq_k
            label = "Ankle" if col == 0 else "Knee"
            ax.plot(t_s, sig, color=C["motor"], lw=1.5)
            style_ax(ax, "Tempo (s)", "Angolo motore (rad)", f"Angolo Motore SEA — {label}")
        else:
            ax.text(0.5, 0.5, "states.sto non disponibile",
                    ha="center", va="center", transform=ax.transAxes,
                    color="gray", fontsize=11)
            ax.set_facecolor("white")
            for sp in ax.spines.values(): sp.set_edgecolor("#cccccc")

    # Riga 4 — Velocità motore
    for ax, col in [(axes[3][0], 0), (axes[3][1], 1)]:
        if has_states:
            sig   = mu_a if col == 0 else mu_k
            label = "Ankle" if col == 0 else "Knee"
            ax.plot(t_s, sig, color=C["motor"], lw=1.5)
            style_ax(ax, "Tempo (s)", "Vel. motore (rad/s)", f"Vel. Motore SEA — {label}")
        else:
            ax.text(0.5, 0.5, "states.sto non disponibile",
                    ha="center", va="center", transform=ax.transAxes,
                    color="gray", fontsize=11)
            ax.set_facecolor("white")
            for sp in ax.spines.values(): sp.set_edgecolor("#cccccc")

    # Riga 5 — Angolo motore + angolo joint sovrapposti
    for ax, col in [(axes[4][0], 0), (axes[4][1], 1)]:
        label = "Ankle" if col == 0 else "Knee"
        q_sig = q_a if col == 0 else q_k
        color = C["ankle_sea"] if col == 0 else C["knee_sea"]
        ax.plot(t_q, q_sig, color=color, lw=1.8, label="Angolo Joint (°)")
        if has_states:
            m_sig = mq_a if col == 0 else mq_k
            # converti rad→° per confronto visivo
            ax.plot(t_s, np.degrees(m_sig), color=C["motor"], lw=1.5,
                    ls="--", alpha=0.9, label="Angolo Motore (°)")
        style_ax(ax, "Tempo (s)", "Angolo (°)", f"Motor vs Joint — {label}")
        add_legend(ax)

    _col_labels(fig, ["Caviglia (Ankle)", "Ginocchio (Knee)"],
                [C["ankle_sea"], C["knee_sea"]])
    fig.suptitle("Cinematica Joint & Motore SEA — vs Tempo",
                 color="black", fontsize=13, fontweight="bold", y=1.02)
    _save_and_show(fig, out_dir, "plot2_motor_kinematics.png")


# ─────────────────────────────────────────────────────────────────────────────
# GRAFICO 3 — 2×2 — Coppia su angolo + Power su GC
# ─────────────────────────────────────────────────────────────────────────────

def plot_torque_angle_power(
    kin_q_df,
    kin_u_df,
    act_df,
    cfg: dict,
    out_dir: str,
    # Dati sani (opzionali)
    kin_q_san=None,
    kin_u_san=None,
    act_san=None,
) -> None:
    """
    Subplot 2×2 — su GC (media ± SD):
      Riga 1: Coppia vs Angolo (loop)
      Riga 2: Potenza vs % Gait Cycle

    Se PLOT_HEALTHY=True in cfg e i DataFrame sani sono forniti,
    sovrappone i dati sani in arancione/verde.
    """
    print("[3/4] plot_torque_angle_power ...")

    ch      = cfg["channels"]
    mass    = cfg["mass"]
    n_pts   = cfg.get("n_pts", 1000)
    plot_h  = cfg.get("PLOT_HEALTHY", False) and all(
        x is not None for x in [kin_q_san, kin_u_san, act_san])

    pct = np.linspace(0, 100, n_pts)

    joints = [
        ("Ankle", ch["joint_ankle_q"], ch["joint_ankle_u"], ch["sea_ankle"],
         ch["res_ankle"], C["ankle_sea"], C["ankle_san"]),
        ("Knee",  ch["joint_knee_q"],  ch["joint_knee_u"],  ch["sea_knee"],
         ch["res_knee"],  C["knee_sea"],  C["knee_san"]),
    ]

    fig, axes = _new_fig(2, 2, (14, 11))

    for col, (joint, q_col, u_col, sea_col, res_col, c_sea, c_san) in enumerate(joints):
        # IC e segnali SEA
        IC_sea = detect_IC(kin_q_df)
        q_sea  = -kin_q_df[q_col].values
        u_sea  = kin_u_df[u_col].values          # segno originale per P
        tau_sea = act_df[sea_col].values
        pwr_sea = compute_power(tau_sea, u_sea, mass)

        pct_q, q_mean, q_std = cycle_stats(q_sea, IC_sea, n_pts)
        pct_p, p_mean, p_std = cycle_stats(pwr_sea, IC_sea, n_pts)

        # Cicli per il loop coppia-angolo
        tau_cycles = get_all_cycles(tau_sea, IC_sea, n_pts)
        q_cycles   = get_all_cycles(q_sea,  IC_sea, n_pts)

        # ── Riga 1: Coppia vs Angolo ─────────────────────────────────────────
        ax_top = axes[0][col]
        ax_top.set_facecolor("white")

        # Cicli individuali SEA
        for q_c, t_c in zip(q_cycles, tau_cycles):
            ax_top.plot(q_c, t_c, color=c_sea, lw=0.8, alpha=0.22)
        # Media SEA
        tau_mean = np.array([get_all_cycles(tau_sea, IC_sea, n_pts)]).mean(0).mean(0) \
            if False else tau_cycles.mean(0)
        ax_top.plot(q_mean, tau_mean, color=c_sea, lw=2.5, label="SEA media")
        ax_top.fill_between(
            q_mean,
            tau_mean - tau_cycles.std(0),
            tau_mean + tau_cycles.std(0),
            color=c_sea, alpha=0.15, label="±1 SD")
        ax_top.scatter(q_mean[0], tau_mean[0], color=c_sea, s=70,
                       zorder=5, edgecolors="black", lw=0.6, label="IC")

        # Dati sani sovrapposti
        if plot_h:
            IC_san  = detect_IC(kin_q_san)
            q_san   = -kin_q_san[q_col].values
            tau_san = act_san[res_col].values
            q_cyc_s  = get_all_cycles(q_san,   IC_san, n_pts)
            t_cyc_s  = get_all_cycles(tau_san, IC_san, n_pts)
            ax_top.plot(q_cyc_s.mean(0), t_cyc_s.mean(0),
                        color=c_san, lw=2.5, ls="--", label="Sano media")
            ax_top.fill_between(q_cyc_s.mean(0),
                                t_cyc_s.mean(0) - t_cyc_s.std(0),
                                t_cyc_s.mean(0) + t_cyc_s.std(0),
                                color=c_san, alpha=0.12)

        ax_top.axhline(0, color=C["zero"], lw=0.8, ls="--")
        ax_top.axvline(0, color=C["zero"], lw=0.8, ls="--")
        style_ax(ax_top, f"Angolo {joint} (°)  [fless. +]",
                 f"Coppia SEA {joint} (Nm)",
                 f"Coppia–Angolo — {joint}")
        add_legend(ax_top)

        # ── Riga 2: Potenza vs % GC ──────────────────────────────────────────
        ax_bot = axes[1][col]
        p_cycs = get_all_cycles(pwr_sea, IC_sea, n_pts)
        _plot_cycle_mean(ax_bot, pct, p_mean, p_std, p_cycs, c_sea)
        _power_fill(ax_bot, pct, p_mean)

        if plot_h:
            IC_san   = detect_IC(kin_q_san)
            u_san    = kin_u_san[u_col].values
            tau_san  = act_san[res_col].values
            pwr_san  = compute_power(tau_san, u_san, mass)
            pm_s, ps_s = get_all_cycles(pwr_san, IC_san, n_pts).mean(0), \
                         get_all_cycles(pwr_san, IC_san, n_pts).std(0)
            ax_bot.fill_between(pct, pm_s - ps_s, pm_s + ps_s, color=c_san, alpha=0.15)
            ax_bot.plot(pct, pm_s, color=c_san, lw=2.5, ls="--", label="Sano media")

        add_gc_events(ax_bot)
        style_ax(ax_bot, "Gait Cycle (%)", f"Potenza SEA {joint} (W/kg)",
                 f"Potenza — {joint}", xlim=(0, 100))
        add_legend(ax_bot)

    _col_labels(fig, ["Caviglia (Ankle)", "Ginocchio (Knee)"],
                [C["ankle_sea"], C["knee_sea"]])
    suffix = "_con_sano" if plot_h else ""
    fig.suptitle(f"Coppia–Angolo & Potenza su GC — Media ± SD{' | + Sano' if plot_h else ''}",
                 color="black", fontsize=13, fontweight="bold", y=1.03)
    _save_and_show(fig, out_dir, f"plot3_torque_angle_power{suffix}.png")


# ─────────────────────────────────────────────────────────────────────────────
# GRAFICO 4 — 3×2 — Angolo, Velocità, Potenza su GC
# ─────────────────────────────────────────────────────────────────────────────

def plot_kinematics_power(
    kin_q_df,
    kin_u_df,
    act_df,
    cfg: dict,
    out_dir: str,
    # Dati sani (opzionali)
    kin_q_san=None,
    kin_u_san=None,
    act_san=None,
) -> None:
    """
    Subplot 3×2 — su GC (media ± SD):
      Riga 1: Angolo joint
      Riga 2: Velocità angolare joint
      Riga 3: Potenza SEA

    Se PLOT_HEALTHY=True in cfg sovrappone i dati sani.
    """
    print("[4/4] plot_kinematics_power ...")

    ch     = cfg["channels"]
    mass   = cfg["mass"]
    n_pts  = cfg.get("n_pts", 1000)
    plot_h = cfg.get("PLOT_HEALTHY", False) and all(
        x is not None for x in [kin_q_san, kin_u_san, act_san])

    pct = np.linspace(0, 100, n_pts)
    IC_sea = detect_IC(kin_q_df)

    joints = [
        ("Ankle", ch["joint_ankle_q"], ch["joint_ankle_u"], ch["sea_ankle"],
         ch["res_ankle"], C["ankle_sea"], C["ankle_san"]),
        ("Knee",  ch["joint_knee_q"],  ch["joint_knee_u"],  ch["sea_knee"],
         ch["res_knee"],  C["knee_sea"],  C["knee_san"]),
    ]

    fig, axes = _new_fig(3, 2, (14, 14))

    for col, (joint, q_col, u_col, sea_col, res_col, c_sea, c_san) in enumerate(joints):
        # Segnali SEA
        q_sea   = -kin_q_df[q_col].values          # flessione +
        u_sea   = -kin_u_df[u_col].values           # flessione +
        tau_sea =  act_df[sea_col].values
        u_raw   =  kin_u_df[u_col].values           # segno originale per P
        pwr_sea = compute_power(tau_sea, u_raw, mass)

        q_cycs  = get_all_cycles(q_sea,   IC_sea, n_pts)
        u_cycs  = get_all_cycles(u_sea,   IC_sea, n_pts)
        p_cycs  = get_all_cycles(pwr_sea, IC_sea, n_pts)

        # Sani
        if plot_h:
            IC_san  = detect_IC(kin_q_san)
            q_san   = -kin_q_san[q_col].values
            u_san_f = -kin_u_san[u_col].values
            tau_san =  act_san[res_col].values
            u_raw_s =  kin_u_san[u_col].values
            pwr_san = compute_power(tau_san, u_raw_s, mass)
            q_s_c   = get_all_cycles(q_san,   IC_san, n_pts)
            u_s_c   = get_all_cycles(u_san_f, IC_san, n_pts)
            p_s_c   = get_all_cycles(pwr_san, IC_san, n_pts)

        # ── Riga 1: Angolo ───────────────────────────────────────────────────
        ax = axes[0][col]
        _plot_cycle_mean(ax, pct, q_cycs.mean(0), q_cycs.std(0), q_cycs, c_sea)
        if plot_h:
            ax.fill_between(pct, q_s_c.mean(0)-q_s_c.std(0),
                            q_s_c.mean(0)+q_s_c.std(0), color=c_san, alpha=0.15)
            ax.plot(pct, q_s_c.mean(0), color=c_san, lw=2.5, ls="--", label="Sano media")
        add_gc_events(ax)
        style_ax(ax, "Gait Cycle (%)", "Angolo (°)  [fless. +]",
                 f"Angolo Joint — {joint}", xlim=(0, 100))
        add_legend(ax)

        # ── Riga 2: Velocità ─────────────────────────────────────────────────
        ax = axes[1][col]
        _plot_cycle_mean(ax, pct, u_cycs.mean(0), u_cycs.std(0), u_cycs, c_sea)
        if plot_h:
            ax.fill_between(pct, u_s_c.mean(0)-u_s_c.std(0),
                            u_s_c.mean(0)+u_s_c.std(0), color=c_san, alpha=0.15)
            ax.plot(pct, u_s_c.mean(0), color=c_san, lw=2.5, ls="--", label="Sano media")
        add_gc_events(ax)
        style_ax(ax, "Gait Cycle (%)", "Vel. Ang. (°/s)  [fless. +]",
                 f"Velocità Angolare Joint — {joint}", xlim=(0, 100))
        add_legend(ax)

        # ── Riga 3: Potenza ──────────────────────────────────────────────────
        ax = axes[2][col]
        _plot_cycle_mean(ax, pct, p_cycs.mean(0), p_cycs.std(0), p_cycs, c_sea)
        _power_fill(ax, pct, p_cycs.mean(0))
        if plot_h:
            ax.fill_between(pct, p_s_c.mean(0)-p_s_c.std(0),
                            p_s_c.mean(0)+p_s_c.std(0), color=c_san, alpha=0.15)
            ax.plot(pct, p_s_c.mean(0), color=c_san, lw=2.5, ls="--", label="Sano media")
        add_gc_events(ax)
        style_ax(ax, "Gait Cycle (%)", "Potenza (W/kg)",
                 f"Potenza SEA — {joint}", xlim=(0, 100))
        add_legend(ax)

    _col_labels(fig, ["Caviglia (Ankle)", "Ginocchio (Knee)"],
                [C["ankle_sea"], C["knee_sea"]])
    suffix = "_con_sano" if plot_h else ""
    fig.suptitle(f"Cinematica & Potenza su GC — Media ± SD{' | + Sano' if plot_h else ''}",
                 color="black", fontsize=13, fontweight="bold", y=1.03)
    _save_and_show(fig, out_dir, f"plot4_kinematics_power{suffix}.png")
