"""
utils.py
========
Funzioni ausiliarie per la lettura e il processing dei file OpenSim .sto.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# ── Lettura file ─────────────────────────────────────────────────────────────

def read_sto(filepath: str) -> pd.DataFrame:
    """
    Legge un file .sto di OpenSim saltando l'header fino a 'endheader'.
    Restituisce un DataFrame con colonne strip-pate.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip() == "endheader":
            start = i + 1
            break
    df = pd.read_csv(filepath, sep="\t", skiprows=start)
    df.columns = df.columns.str.strip()
    return df


def try_read_sto(filepath: str) -> pd.DataFrame | None:
    """
    Come read_sto ma restituisce None se il file non esiste,
    stampando un avviso invece di sollevare un'eccezione.
    """
    import os
    if not os.path.isfile(filepath):
        print(f"[AVVISO] File non trovato, subplot saltati: {filepath}")
        return None
    return read_sto(filepath)


def check_columns(df: pd.DataFrame, required: set, filename: str = "") -> None:
    """Verifica che il DataFrame contenga tutte le colonne richieste."""
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Colonne mancanti in '{filename}': {missing}")


# ── Gait cycle ───────────────────────────────────────────────────────────────

def detect_IC(kin_q: pd.DataFrame, knee_col: str = "pros_knee_angle",
              min_cycle_s: float = 0.85) -> np.ndarray:
    """
    Rileva gli Initial Contact (IC) come massimi di pros_knee_angle
    (ginocchio più esteso ≈ 0°), con distanza minima min_cycle_s secondi.

    Parametri
    ---------
    kin_q       : DataFrame con colonna temporale 'time' e knee_col
    knee_col    : nome della colonna angolo ginocchio (default: pros_knee_angle)
    min_cycle_s : distanza minima in secondi tra due IC consecutivi

    Restituisce
    -----------
    Array di indici degli IC nel DataFrame.
    """
    t = kin_q["time"].values
    dt = float(np.mean(np.diff(t)))
    knee = kin_q[knee_col].values
    peaks, _ = find_peaks(knee, prominence=3, distance=int(min_cycle_s / dt))
    if len(peaks) < 2:
        raise ValueError(
            f"Trovati solo {len(peaks)} IC: impossibile estrarre gait cycle completi."
        )
    return peaks


def interpolate_cycle(signal: np.ndarray, i0: int, i1: int,
                      n_pts: int = 1000) -> np.ndarray:
    """Interpola un singolo ciclo [i0:i1] su una griglia di n_pts punti 0–100%."""
    seg = signal[i0:i1]
    pct_src = np.linspace(0, 100, len(seg))
    pct_dst = np.linspace(0, 100, n_pts)
    return np.interp(pct_dst, pct_src, seg)


def get_all_cycles(signal: np.ndarray, IC: np.ndarray,
                   n_pts: int = 1000) -> np.ndarray:
    """
    Ritaglia e interpola tutti i cicli definiti da IC su griglia comune 0–100%.

    Restituisce array shape (n_cicli, n_pts).
    """
    cycles = []
    for i in range(len(IC) - 1):
        cycles.append(interpolate_cycle(signal, IC[i], IC[i + 1] + 1, n_pts))
    return np.array(cycles)


def cycle_stats(signal: np.ndarray, IC: np.ndarray,
                n_pts: int = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcola media e deviazione standard su tutti i gait cycle.

    Restituisce (pct_grid, mean, std).
    """
    all_c = get_all_cycles(signal, IC, n_pts)
    return (np.linspace(0, 100, n_pts), all_c.mean(axis=0), all_c.std(axis=0))


# ── Potenza ──────────────────────────────────────────────────────────────────

def compute_power(torque: np.ndarray, velocity_degs: np.ndarray,
                  mass: float) -> np.ndarray:
    """
    Calcola la potenza normalizzata per la massa corporea.

    P [W/kg] = torque [Nm] × velocity [rad/s] / mass [kg]
    La velocity viene convertita da °/s a rad/s internamente.
    """
    DEG2RAD = np.pi / 180.0
    return torque * (velocity_degs * DEG2RAD) / mass


# ── Helpers per plotting ─────────────────────────────────────────────────────

GC_EVENTS = [
    (0,  "IC"),
    (12, "LR"),
    (31, "Mid-St"),
    (50, "Pre-Sw"),
    (60, "TO"),
    (73, "Sw pk"),
]


def add_gc_events(ax) -> None:
    """Aggiunge linee verticali etichettate per gli eventi del gait cycle."""
    for perc, lbl in GC_EVENTS:
        ax.axvline(perc, color="#cccccc", lw=0.9, ls="--")
        ax.text(perc + 0.5, 0.97, lbl, fontsize=7, color="#888",
                rotation=90, va="top", transform=ax.get_xaxis_transform())


def style_ax(ax, xlabel: str, ylabel: str, title: str,
             xlim: tuple | None = None) -> None:
    """Applica stile uniforme a un Axes."""
    ax.axhline(0, color="#aaaaaa", lw=0.8, ls="--")
    ax.set_xlabel(xlabel, color="black", fontsize=10)
    ax.set_ylabel(ylabel, color="black", fontsize=10)
    ax.set_title(title, color="black", fontsize=11, fontweight="bold", pad=5)
    ax.tick_params(colors="black")
    ax.set_facecolor("white")
    if xlim is not None:
        ax.set_xlim(xlim)
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc")


def add_legend(ax, **kwargs) -> None:
    """Aggiunge legenda con stile uniforme."""
    ax.legend(facecolor="white", edgecolor="#cccccc",
              labelcolor="black", fontsize=9,
              loc=kwargs.get("loc", "upper right"))
