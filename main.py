"""
main.py
=======
File principale — sezione di configurazione + chiamata a tutte le funzioni plotter.

STRUTTURA DEL PROGETTO
----------------------
  main.py        ← questo file (config + orchestrazione)
  utils.py       ← funzioni ausiliarie (lettura .sto, IC, interpolazione, potenza)
  plotters.py    ← funzioni plotter (una per grafico)

GRAFICI GENERATI
----------------
  1. plot1_torque_time.png          — 4×2 — SEA torque, control, reserve vs tempo
  2. plot2_motor_kinematics.png     — 5×2 — angoli e velocità joint e motore vs tempo
  3. plot3_torque_angle_power.png   — 2×2 — coppia/angolo e power su GC
  4. plot4_kinematics_power.png     — 3×2 — angolo, velocità, power su GC
  (3 e 4 hanno anche versione "_con_sano" se PLOT_HEALTHY = True)
"""

import os
import sys

# Rendi importabili utils e plotters dalla stessa cartella
sys.path.insert(0, os.path.dirname(__file__))

from utils   import read_sto, try_read_sto, check_columns
from plotters import (
    plot_sea_torque_time,
    plot_motor_kinematics,
    plot_torque_angle_power,
    plot_kinematics_power,
)

# ══════════════════════════════════════════════════════════════════════════════
#  SEZIONE DI CONFIGURAZIONE  ← modifica solo questa sezione
# ══════════════════════════════════════════════════════════════════════════════

# ── Directory base dei file SEA (protesi) ─────────────────────────────────────
BASE_SEA = r"C:\Users\tomma\Desktop\Opensim OMNIBUS\21_76-WellScaled-SEA\CMC\Risultati_SEASEA-impedence"

# ── File SEA ─────────────────────────────────────────────────────────────────
FILES_SEA = {
    "kin_q":   os.path.join(BASE_SEA, "3DGaitModel2392_Kinematics_q.sto"),
    "kin_u":   os.path.join(BASE_SEA, "3DGaitModel2392_Kinematics_u.sto"),
    "act":     os.path.join(BASE_SEA, "3DGaitModel2392_Actuation_force.sto"),
    "ctrl":    os.path.join(BASE_SEA, "3DGaitModel2392_controls.sto"),
    # Se non disponibile lascia il percorso vuoto ("") → i subplot saranno saltati
    "states":  os.path.join(BASE_SEA, "3DGaitModel2392_states.sto"),
}

# ── Directory base dei file sani ─────────────────────────────────────────────
BASE_SAN = r"C:\Users\tomma\Desktop\Opensim OMNIBUS\3D_Model_Leg_and_Prosthesis_Completo_21_76\CMC\Risultati(IKResults)"

# ── File sani ─────────────────────────────────────────────────────────────────
FILES_SAN = {
    "kin_q":   os.path.join(BASE_SAN, "3DGaitModel2392_Kinematics_q.sto"),
    "kin_u":   os.path.join(BASE_SAN, "3DGaitModel2392_Kinematics_u.sto"),
    "act":     os.path.join(BASE_SAN, "3DGaitModel2392_Actuation_force.sto"),
}

# ── Directory di output ───────────────────────────────────────────────────────
OUT_DIR = os.path.join(BASE_SEA, "Grafici")

# ── Nomi dei canali ──────────────────────────────────────────────────────────
CHANNELS = {
    # Cinematica joint (stessi nomi in SEA e sano)
    "joint_ankle_q":    "pros_ankle_angle",
    "joint_knee_q":     "pros_knee_angle",
    "joint_ankle_u":    "pros_ankle_angle",
    "joint_knee_u":     "pros_knee_angle",

    # Attuatori SEA (da Actuation_force SEA)
    "sea_ankle":        "SEA_Ankle",
    "sea_knee":         "SEA_Knee",

    # Control input (da controls.sto SEA)
    "ctrl_ankle":       "SEA_Ankle",
    "ctrl_knee":        "SEA_Knee",

    # Reserve actuators (da Actuation_force)
    "res_ankle":        "reserve_pros_ankle_angle",
    "res_knee":         "reserve_pros_knee_angle",

    # Angolo e velocità motore (da states.sto — nomi canale OpenSim)
    "motor_angle_ankle": "/forceset/SEA_Ankle/motor_angle",
    "motor_angle_knee":  "/forceset/SEA_Knee/motor_angle",
    "motor_speed_ankle": "/forceset/SEA_Ankle/motor_speed",
    "motor_speed_knee":  "/forceset/SEA_Knee/motor_speed",
}

# ── Costanti fisiche ──────────────────────────────────────────────────────────
MASS = 76.0          # kg — massa corporea del soggetto

# ── Opzioni ──────────────────────────────────────────────────────────────────
PLOT_HEALTHY = True  # True = sovrappone dati sani nei grafici 3 e 4
N_PTS        = 1000  # punti di interpolazione per il gait cycle

# ── Selezione grafici da generare ─────────────────────────────────────────────
# Imposta a False per saltare un grafico
RUN = {
    "plot1": True,
    "plot2": True,
    "plot3": True,
    "plot4": True,
}

# ══════════════════════════════════════════════════════════════════════════════
#  FINE SEZIONE DI CONFIGURAZIONE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Configurazione unificata ──────────────────────────────────────────────
    cfg = {
        "channels":     CHANNELS,
        "mass":         MASS,
        "n_pts":        N_PTS,
        "PLOT_HEALTHY": PLOT_HEALTHY,
    }

    # ── Caricamento file SEA ──────────────────────────────────────────────────
    print("=== Caricamento file SEA ===")
    kin_q = read_sto(FILES_SEA["kin_q"])
    kin_u = read_sto(FILES_SEA["kin_u"])
    act   = read_sto(FILES_SEA["act"])
    ctrl  = read_sto(FILES_SEA["ctrl"])

    # States: opzionale
    states_path = FILES_SEA.get("states", "")
    states = try_read_sto(states_path) if states_path else None

    # Se states disponibile, adatta i nomi canale (potrebbero variare)
    if states is not None:
        # Verifica che i canali motore esistano
        motor_cols = {
            CHANNELS["motor_angle_ankle"], CHANNELS["motor_angle_knee"],
            CHANNELS["motor_speed_ankle"], CHANNELS["motor_speed_knee"],
        }
        missing = motor_cols - set(states.columns)
        if missing:
            print(f"[AVVISO] Canali motore mancanti in states.sto: {missing}")
            print("  → Grafici motore saltati.")
            states = None

    # ── Caricamento file sani ─────────────────────────────────────────────────
    kin_q_san = kin_u_san = act_san = None
    if PLOT_HEALTHY:
        print("\n=== Caricamento file sani ===")
        try:
            kin_q_san = read_sto(FILES_SAN["kin_q"])
            kin_u_san = read_sto(FILES_SAN["kin_u"])
            act_san   = read_sto(FILES_SAN["act"])
        except FileNotFoundError as e:
            print(f"[AVVISO] File sano non trovato: {e}")
            print("  → PLOT_HEALTHY disabilitato.")
            kin_q_san = kin_u_san = act_san = None

    print(f"\n=== Output in: {OUT_DIR} ===\n")

    # ── Grafico 1 ─────────────────────────────────────────────────────────────
    if RUN["plot1"]:
        plot_sea_torque_time(
            act_df=act, ctrl_df=ctrl,
            cfg=cfg, out_dir=OUT_DIR,
        )

    # ── Grafico 2 ─────────────────────────────────────────────────────────────
    if RUN["plot2"]:
        plot_motor_kinematics(
            kin_q_df=kin_q, kin_u_df=kin_u, states_df=states,
            cfg=cfg, out_dir=OUT_DIR,
        )

    # ── Grafico 3 ─────────────────────────────────────────────────────────────
    if RUN["plot3"]:
        plot_torque_angle_power(
            kin_q_df=kin_q, kin_u_df=kin_u, act_df=act,
            cfg=cfg, out_dir=OUT_DIR,
            kin_q_san=kin_q_san, kin_u_san=kin_u_san, act_san=act_san,
        )

    # ── Grafico 4 ─────────────────────────────────────────────────────────────
    if RUN["plot4"]:
        plot_kinematics_power(
            kin_q_df=kin_q, kin_u_df=kin_u, act_df=act,
            cfg=cfg, out_dir=OUT_DIR,
            kin_q_san=kin_q_san, kin_u_san=kin_u_san, act_san=act_san,
        )

    print("\n=== Tutti i grafici completati ===")


if __name__ == "__main__":
    main()
