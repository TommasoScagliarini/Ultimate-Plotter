"""
SEA Parameter Optimization for CMC Simulation
==============================================
Phase 1: Coarse grid search (parallel)
Phase 2: Differential evolution refinement (parallel)
Phase 3: Results analysis and export

NOTE: CMC non gestisce spazi nei path letti dall'XML.
      Tutti i file temporanei vengono copiati in WORK_DIR (senza spazi).
      I file XML copiati vengono anche "patchati" internamente per
      risolvere eventuali path con spazi contenuti dentro di essi
      (es. path GRF dentro Externall_Loads.xml).

      Su Windows, differential_evolution con workers=N usa spawn e i
      globali Python non vengono ereditati dai processi figli. Il
      support_mapping viene quindi salvato su disco come JSON e
      ricaricato da ogni worker della Phase 2.
"""

import os
import json
import shutil
import subprocess
import multiprocessing as mp
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import itertools
import time
import signal
import threading
from scipy.optimize import differential_evolution

# ==============================================================================
# 1. CONFIGURAZIONE PERCORSI
# ==============================================================================

BASE_DIR             = r"C:\Users\tomma\Desktop\Opensim OMNIBUS\21_76-WellScaled-SEA\CMC"
BASE_SAN             = r"C:\Users\tomma\Desktop\Opensim OMNIBUS\3D_Model_Leg_and_Prosthesis_Completo_21_76\CMC\Risultati(IKResults)"
SETUP_FILE_BASE      = os.path.join(BASE_DIR, "CMC_Setup_SEASEA-impedence.xml")
MODEL_FILE_BASE      = r"C:\Users\tomma\Desktop\Opensim OMNIBUS\21_76-WellScaled-SEA\Adjusted_SEASEA.osim"
REFERENCE_KINEMATICS = os.path.join(BASE_SAN, "3DGaitModel2392_Kinematics_q.sto")
CMC_EXE              = r"C:\OpenSim-mCMC\bin\cmc.exe"
PLUGIN_DLL           = r"C:\OpenSim-mCMC\plugins\SEA_Plugin_BlackBox_mCMC_impedence.dll"

# Directory di lavoro SENZA SPAZI
WORK_DIR       = r"C:\CMC_Sweep"
RESULTS_DIR    = os.path.join(WORK_DIR, "sweep_results")
SUMMARY_CSV    = os.path.join(WORK_DIR, "sweep_summary_global.csv")
MAPPING_CACHE  = os.path.join(WORK_DIR, "_support_mapping.json")  # per Phase 2

TARGET_ACTUATORS = ["SEA_Ankle", "SEA_Knee"]

# Mapping colonne: file simulato -> riferimento sano
JOINT_COLS_MAP = {
    "pros_ankle_angle": "pros_ankle_angle",
    "pros_knee_angle":  "pros_knee_angle",
}

# Tag del setup CMC che referenziano file su disco
SETUP_FILE_TAGS = [
    'coordinates_file',
    'desired_kinematics_file',
    'task_set_file',
    'constraint_set_file',
    'force_set_files',
    'external_loads_file',
    'actuator_set_files',
    'control_constraints_file',
]

# ==============================================================================
# 2. COPIA E PATCH DEI FILE DI SUPPORTO IN WORK_DIR
# ==============================================================================

def _patch_xml_internal_paths(xml_path):
    """
    Apre un file XML gia' copiato in WORK_DIR e sostituisce tutti i path
    assoluti con spazi con copie in WORK_DIR. Salva su se stesso.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        modified = False
        for elem in root.iter():
            if not elem.text or not elem.text.strip():
                continue
            text = elem.text.strip()
            if os.path.isabs(text) and ' ' in text and os.path.exists(text):
                dst = os.path.join(WORK_DIR, os.path.basename(text))
                shutil.copy2(text, dst)
                elem.text = dst
                modified = True
                print(f"    [PATCH] {os.path.basename(xml_path)}: "
                      f"'{os.path.basename(text)}' -> WORK_DIR")
        if modified:
            tree.write(xml_path, encoding='unicode', xml_declaration=False)
    except Exception as e:
        print(f"  [WARN] _patch_xml_internal_paths({xml_path}): {e}")


def copy_support_files_to_workdir():
    """
    Copia in WORK_DIR tutti i file referenziati nel setup CMC e il modello.
    Patcha i path interni degli XML copiati.
    Restituisce {path_originale: path_in_workdir} e lo salva su MAPPING_CACHE.
    """
    os.makedirs(WORK_DIR, exist_ok=True)
    mapping = {}

    try:
        tree = ET.parse(SETUP_FILE_BASE)
        root = tree.getroot()
        cmc  = root.find('.//CMCTool')
        if cmc is None:
            print("  [WARN] Tag <CMCTool> non trovato nel setup.")
            return mapping

        base = os.path.dirname(os.path.abspath(SETUP_FILE_BASE))

        for tag in SETUP_FILE_TAGS:
            node = cmc.find(f'.//{tag}')
            if node is None or not node.text or not node.text.strip():
                continue
            text = node.text.strip()
            if text.lower() in ('none', ''):
                continue
            src = text if os.path.isabs(text) \
                       else os.path.normpath(os.path.join(base, text))
            if os.path.exists(src):
                dst = os.path.join(WORK_DIR, os.path.basename(src))
                shutil.copy2(src, dst)
                mapping[src] = dst
                print(f"  [COPY] {os.path.basename(src)}")
            else:
                print(f"  [WARN] File non trovato: {src}  (tag: <{tag}>)")
    except Exception as e:
        print(f"  [WARN] copy_support_files_to_workdir: {e}")

    # Copia modello base
    if os.path.exists(MODEL_FILE_BASE):
        dst_model = os.path.join(WORK_DIR, os.path.basename(MODEL_FILE_BASE))
        shutil.copy2(MODEL_FILE_BASE, dst_model)
        mapping[MODEL_FILE_BASE] = dst_model
        print(f"  [COPY] {os.path.basename(MODEL_FILE_BASE)}")

    # Patcha path interni degli XML
    print("  Patching path interni degli XML...")
    for fname in os.listdir(WORK_DIR):
        if fname.lower().endswith('.xml'):
            _patch_xml_internal_paths(os.path.join(WORK_DIR, fname))

    # Salva mapping su disco per la Phase 2 (spawn su Windows)
    with open(MAPPING_CACHE, 'w') as f:
        json.dump(mapping, f, indent=2)

    return mapping


def load_mapping():
    """Carica il support_mapping da MAPPING_CACHE (usato dai worker Phase 2)."""
    with open(MAPPING_CACHE, 'r') as f:
        return json.load(f)


# ==============================================================================
# 3. LETTURA FILE STO
# ==============================================================================

def _read_sto(filepath):
    """
    Legge un file .sto/.mot di OpenSim.
    Trova 'endheader', poi cerca la prima riga con tab come
    intestazione colonne reale (salta righe spurie come 'time', 'header').
    """
    with open(filepath, 'r', errors='replace') as f:
        lines = f.readlines()

    after_endheader = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == 'endheader':
            after_endheader = i + 1
            break

    header_row = after_endheader
    for i in range(after_endheader, len(lines)):
        if '\t' in lines[i]:
            header_row = i
            break

    df = pd.read_csv(filepath, sep='\t', skiprows=header_row)
    df.columns = df.columns.str.strip()
    return df


# ==============================================================================
# 4. COST FUNCTION
# ==============================================================================

def evaluate_run_cost(results_dir):
    """
    Cost = W_KIN  * sum((q_sim - q_ref)^2)
         + W_CHAT * sum(diff(omega_motor)^2)
    Restituisce 1e9 se la simulazione e' fallita.
    """
    try:
        # ── Errore cinematico ─────────────────────────────────────────────────
        kin_cost = 0.0
        if os.path.exists(REFERENCE_KINEMATICS):
            kin_files = [f for f in os.listdir(results_dir)
                         if f.endswith('_Kinematics_q.sto')]
            if not kin_files:
                return 1e9

            df_sim = _read_sto(os.path.join(results_dir, kin_files[0]))
            df_ref = _read_sto(REFERENCE_KINEMATICS)
            t_sim  = df_sim['time'].values
            t_ref  = df_ref['time'].values

            for sim_col, ref_col in JOINT_COLS_MAP.items():
                if sim_col not in df_sim.columns:
                    print(f"  [WARN] Colonna '{sim_col}' non trovata in Kinematics_q.sto")
                    continue
                if ref_col not in df_ref.columns:
                    print(f"  [WARN] Colonna '{ref_col}' non trovata nel riferimento sano")
                    continue
                ref_interp = np.interp(t_sim, t_ref, df_ref[ref_col].values)
                kin_cost += float(np.sum((df_sim[sim_col].values - ref_interp) ** 2))

        # ── Chattering velocità motore ────────────────────────────────────────
        chatter_cost = 0.0
        states_files = [f for f in os.listdir(results_dir)
                        if f.endswith('_states.sto')]
        if states_files:
            df_states = _read_sto(os.path.join(results_dir, states_files[0]))
            speed_cols = [c for c in df_states.columns if 'motor_speed' in c.lower()]
            if not speed_cols:
                print(f"  [WARN] Nessuna colonna 'motor_speed' in _states.sto")
                print(f"         Colonne: {list(df_states.columns)}")
            else:
                for col in speed_cols:
                    chatter_cost += float(np.sum(np.diff(df_states[col].values) ** 2))

        W_KIN, W_CHAT = 1.0, 1.0
        total = W_KIN * kin_cost + W_CHAT * chatter_cost
        print(f"  [COST] kin={kin_cost:.2f}  chatter={chatter_cost:.2f}  total={total:.2f}")
        return total

    except Exception as e:
        print(f"  [WARN] evaluate_run_cost: {e}")
        return 1e9


# ==============================================================================
# 5. MODIFICA MODELLO E SETUP
# ==============================================================================

def _set_actuator_param(root, actuator_name, param_name, value):
    for actuator in root.iter('SeriesElasticActuator'):
        if actuator.get('name') == actuator_name:
            node = actuator.find(f'.//{param_name}')
            if node is None:
                print(f"  [ERROR] <{param_name}> non trovato in {actuator_name}")
                return False
            node.text = str(value)
            return True
    print(f"  [ERROR] Attuatore {actuator_name} non trovato nel modello")
    return False


def _resolve_setup_paths(cmc_tool, support_mapping):
    base = os.path.dirname(os.path.abspath(SETUP_FILE_BASE))
    for tag in SETUP_FILE_TAGS:
        node = cmc_tool.find(f'.//{tag}')
        if node is None or not node.text or not node.text.strip():
            continue
        text = node.text.strip()
        if text.lower() in ('none', ''):
            continue
        src = text if os.path.isabs(text) \
                   else os.path.normpath(os.path.join(base, text))
        if src in support_mapping:
            node.text = support_mapping[src]
        else:
            print(f"  [WARN] Nessun mapping trovato per <{tag}>: {src}")


def build_run_files(kp_ankle, kd_ankle, kp_knee, kd_knee,
                    run_dir, run_name, support_mapping):
    os.makedirs(run_dir, exist_ok=True)
    temp_model = os.path.join(run_dir, "temp_model.osim")
    temp_setup = os.path.join(run_dir, "temp_setup.xml")

    model_src = support_mapping.get(MODEL_FILE_BASE, MODEL_FILE_BASE)
    try:
        tree_model = ET.parse(model_src)
    except ET.ParseError as e:
        print(f"  [ERROR] Parsing modello: {e}")
        return None, None

    root_model = tree_model.getroot()
    params = {
        "SEA_Ankle": {"Kp": kp_ankle, "Kd": kd_ankle},
        "SEA_Knee":  {"Kp": kp_knee,  "Kd": kd_knee},
    }
    for actuator_name, pdict in params.items():
        for pname, pval in pdict.items():
            if not _set_actuator_param(root_model, actuator_name, pname, pval):
                return None, None
    tree_model.write(temp_model, encoding='unicode', xml_declaration=False)

    try:
        tree_setup = ET.parse(SETUP_FILE_BASE)
    except ET.ParseError as e:
        print(f"  [ERROR] Parsing setup: {e}")
        return None, None

    root_setup = tree_setup.getroot()
    cmc_tool   = root_setup.find('.//CMCTool')
    if cmc_tool is not None:
        _resolve_setup_paths(cmc_tool, support_mapping)
        for tag, value in [('model_file',        temp_model),
                           ('results_directory', run_dir),
                           ('name',              run_name)]:
            node = cmc_tool.find(f'.//{tag}')
            if node is not None:
                node.text = value
    tree_setup.write(temp_setup, encoding='unicode', xml_declaration=False)

    return temp_model, temp_setup


# ==============================================================================
# 6. SISTEMA DI PROGRESS
# ==============================================================================

def _format_time(seconds):
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m{int(seconds%60):02d}s"
    else:
        return f"{int(seconds//3600)}h{int((seconds%3600)//60):02d}m"


def _print_bar(completed, total, elapsed_times, phase_label):
    pct    = completed / total if total > 0 else 0
    filled = int(pct * 35)
    bar    = '#' * filled + '-' * (35 - filled)
    if elapsed_times and completed > 0:
        avg     = sum(elapsed_times) / len(elapsed_times)
        eta_sec = avg * (total - completed)
        eta_str = _format_time(eta_sec)
        avg_str = _format_time(avg)
    else:
        eta_str = '--:--'
        avg_str = '--'
    print(f"\r  {phase_label}  [{bar}]  {completed}/{total}  "
          f"({pct*100:.0f}%)  media/run: {avg_str}  ETA: {eta_str}   ",
          end='', flush=True)


# ==============================================================================
# 7. WORKER
# ==============================================================================

def run_cmc_worker(params):
    """
    params: (kp_ankle, kd_ankle, kp_knee, kd_knee, run_id, support_mapping)
    Restituisce: (kp_ankle, kd_ankle, kp_knee, kd_knee, cost, elapsed)
    """
    kp_ankle, kd_ankle, kp_knee, kd_knee, run_id, support_mapping = params[:6]
    t0 = time.time()

    run_name = (f"run{run_id:04d}"
                f"_KpK{kp_knee}_KdK{kd_knee}"
                f"_KpA{kp_ankle}_KdA{kd_ankle}")
    run_dir  = os.path.join(RESULTS_DIR, run_name)

    # Salta se gia' eseguita
    existing = [f for f in os.listdir(run_dir)
                if f.endswith('_controls.sto')] \
               if os.path.isdir(run_dir) else []
    if existing:
        cost = evaluate_run_cost(run_dir)
        elapsed = time.time() - t0
        print(f"\n  [SKIP] {run_name} -> costo: {cost:.4f}")
        return (kp_ankle, kd_ankle, kp_knee, kd_knee, cost, elapsed)

    temp_model, temp_setup = build_run_files(
        kp_ankle, kd_ankle, kp_knee, kd_knee,
        run_dir, run_name, support_mapping
    )
    if temp_model is None:
        return (kp_ankle, kd_ankle, kp_knee, kd_knee, 1e9, time.time() - t0)

    cmd = [CMC_EXE, "-Setup", temp_setup, "-Library", PLUGIN_DLL]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=WORK_DIR,
            check=True,
            timeout=1800
        )
        cost = evaluate_run_cost(run_dir)

    except subprocess.TimeoutExpired:
        print(f"\n  [TIMEOUT] {run_name}")
        cost = 1e9

    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode('utf-8', errors='replace') if e.stderr else ""
        stdout_text = e.stdout.decode('utf-8', errors='replace') if e.stdout else ""
        err_lines = [l for l in stderr_text.splitlines() if l.strip()][-20:]
        out_lines = [l for l in stdout_text.splitlines() if l.strip()][-20:]
        print(f"\n  [FAIL] {run_name} — returncode {e.returncode}")
        if err_lines:
            print("  [STDERR]\n    " + "\n    ".join(err_lines))
        if out_lines:
            print("  [STDOUT]\n    " + "\n    ".join(out_lines))
        cost = 1e9

    elapsed = time.time() - t0
    print(f"\n  [{run_id:04d}] Kp Knee:{kp_knee:5d} Kd Knee:{kd_knee:4d} | "
          f"Kp Ankle:{kp_ankle:5d} Kd Ankle:{kd_ankle:4d} -> "
          f"costo: {cost:.4f}  ({elapsed:.1f}s)")

    return (kp_ankle, kd_ankle, kp_knee, kd_knee, cost, elapsed)


# ==============================================================================
# 8. PHASE 1 — GRID SEARCH GROSSOLANO
# ==============================================================================

def phase1_grid_search(num_cores, support_mapping):
    print("\n" + "="*60)
    print("PHASE 1 — Grid Search Grossolano")
    print("="*60)

    grid_kp_ankle = [6000, 7500, 9000]
    grid_kd_ankle = [10,   20,   50]
    grid_kp_knee  = [2000, 2500, 3000]
    grid_kd_knee  = [10,   20,   50]

    combinations = list(itertools.product(
        grid_kp_ankle, grid_kd_ankle,
        grid_kp_knee,  grid_kd_knee
    ))
    tasks = [(*combo, run_id, support_mapping)
             for run_id, combo in enumerate(combinations, start=1)]

    total   = len(tasks)
    results = []
    times   = []
    t_start = time.time()

    print(f"Combinazioni totali: {total}")
    print(f"Core utilizzati:     {num_cores}\n")
    _print_bar(0, total, [], "Phase 1")

    with mp.Pool(num_cores) as pool:
        for result in pool.imap_unordered(run_cmc_worker, tasks):
            results.append(result)
            elapsed_run = result[5] if len(result) > 5 else 0.0
            times.append(elapsed_run)
            _print_bar(len(results), total, times, "Phase 1")

    print()
    print(f"  Completato in {_format_time(time.time() - t_start)}\n")
    return results


# ==============================================================================
# 9. PHASE 2 — DIFFERENTIAL EVOLUTION (raffinamento)
# ==============================================================================

_de_counter = mp.Value('i', 10000)


def _de_cost_wrapper(x):
    """
    Wrapper per differential_evolution.
    Carica il mapping da file JSON — necessario su Windows con spawn,
    dove i globali Python non vengono ereditati dai processi figli.
    """
    kp_ankle = int(round(x[0]))
    kd_ankle = int(round(x[1]))
    kp_knee  = int(round(x[2]))
    kd_knee  = int(round(x[3]))

    with _de_counter.get_lock():
        _de_counter.value += 1
        run_id = _de_counter.value

    # Carica il mapping da disco invece di usare un globale
    support_mapping = load_mapping()

    result = run_cmc_worker(
        (kp_ankle, kd_ankle, kp_knee, kd_knee, run_id, support_mapping)
    )
    return result[4]   # cost è il quinto elemento


def phase2_differential_evolution(best_from_phase1, num_cores):
    print("\n" + "="*60)
    print("PHASE 2 — Differential Evolution (raffinamento)")
    print("="*60)

    kp_a0 = best_from_phase1['Kp_Ankle']
    kd_a0 = best_from_phase1['Kd_Ankle']
    kp_k0 = best_from_phase1['Kp_Knee']
    kd_k0 = best_from_phase1['Kd_Knee']

    def bounded(v, lo, hi):
        return (max(lo, v * 0.5), min(hi, v * 2.0))

    bounds = [
        bounded(kp_a0,  1000, 15000),
        bounded(kd_a0,     5,   300),
        bounded(kp_k0,  1000, 10000),
        bounded(kd_k0,     5,   200),
    ]

    print("Bounds ricavati dal best di Phase 1:")
    for lb, b in zip(["Kp_Ankle", "Kd_Ankle", "Kp_Knee", "Kd_Knee"], bounds):
        print(f"  {lb}: [{b[0]:.0f}, {b[1]:.0f}]")
    print()

    result = differential_evolution(
        _de_cost_wrapper,
        bounds,
        #maxiter=15,
        #popsize=6,
        maxiter=5,
        popsize=4,
        tol=1e-4,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        workers=num_cores,
        polish=False,
        disp=True,
    )

    best_x = result.x
    return {
        'Kp_Ankle': int(round(best_x[0])),
        'Kd_Ankle': int(round(best_x[1])),
        'Kp_Knee':  int(round(best_x[2])),
        'Kd_Knee':  int(round(best_x[3])),
        'Cost':     result.fun,
    }


# ==============================================================================
# 10. ANALISI E SALVATAGGIO RISULTATI
# ==============================================================================

def save_and_print_results(all_results, best_de=None):
    cols = ["Kp_Ankle", "Kd_Ankle", "Kp_Knee", "Kd_Knee", "Cost"]
    # Prendi solo i primi 5 elementi da ogni risultato (escludi elapsed)
    data = [r[:5] for r in all_results]
    df = pd.DataFrame(data, columns=cols)
    df.sort_values('Cost', inplace=True)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nRisultati salvati in: {SUMMARY_CSV}")

    valid = df[df['Cost'] < 1e8]
    if valid.empty:
        print("\n[WARN] Tutte le simulazioni sono fallite.")
        return None

    best_grid = valid.iloc[0]
    print("\n--- Top 5 Grid Search ---")
    print(valid.head(5).to_string(index=False))

    print("\n--- Miglior configurazione Phase 1 (Grid) ---")
    print(f"  Kp Ankle: {best_grid['Kp_Ankle']:.0f}")
    print(f"  Kd Ankle: {best_grid['Kd_Ankle']:.0f}")
    print(f"  Kp Knee:  {best_grid['Kp_Knee']:.0f}")
    print(f"  Kd Knee:  {best_grid['Kd_Knee']:.0f}")
    print(f"  Costo:    {best_grid['Cost']:.4f}")

    if best_de is not None:
        print("\n--- Miglior configurazione Phase 2 (Differential Evolution) ---")
        print(f"  Kp Ankle: {best_de['Kp_Ankle']}")
        print(f"  Kd Ankle: {best_de['Kd_Ankle']}")
        print(f"  Kp Knee:  {best_de['Kp_Knee']}")
        print(f"  Kd Knee:  {best_de['Kd_Knee']}")
        print(f"  Costo:    {best_de['Cost']:.4f}")

        if best_grid['Cost'] > 0:
            improvement = (best_grid['Cost'] - best_de['Cost']) / best_grid['Cost'] * 100
            print(f"\n  Miglioramento rispetto al grid: {improvement:.1f}%")

    return best_grid.to_dict()


# ==============================================================================
# 11. MAIN
# ==============================================================================

if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    os.makedirs(WORK_DIR,    exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    #num_cores = max(1, mp.cpu_count() // 2)
    num_cores = 5

    print("="*60)
    print("SEA Parameter Optimization — CMC Simulation")
    print(f"BASE_DIR:   {BASE_DIR}")
    print(f"WORK_DIR:   {WORK_DIR}  (senza spazi)")
    print(f"Core usati: {num_cores}")
    print("="*60)

    for f, label in [(SETUP_FILE_BASE,      "CMC Setup"),
                     (MODEL_FILE_BASE,       "Modello OSim"),
                     (REFERENCE_KINEMATICS,  "Cinematica riferimento")]:
        status = "OK" if os.path.exists(f) else "NON TROVATO"
        print(f"  [{status}] {label}: {f}")
    print()

    # Copia file di supporto e salva mapping JSON su disco
    print("Copio i file di supporto in WORK_DIR...")
    support_mapping = copy_support_files_to_workdir()
    print(f"  {len(support_mapping)} file copiati.")
    print(f"  Mapping salvato in: {MAPPING_CACHE}\n")

    # ------------------------------------------------------------------
    # TEST SINGOLA RUN — decommenta per diagnosticare prima del sweep
    # ------------------------------------------------------------------
    """
    print("\n--- TEST SINGOLA RUN ---")
    print("  Avvio simulazione...", flush=True)
    t0 = time.time()
    result = run_cmc_worker((7500, 20, 2500, 20, 9999, support_mapping))
    print(f"  Completata in {time.time()-t0:.1f}s  |  costo: {result[4]:.4f}")
    exit(0)
    """

    try:
        # ------------------------------------------------------------------
        # PHASE 1 — Grid search
        # ------------------------------------------------------------------
        grid_results = phase1_grid_search(num_cores, support_mapping)

        cols       = ["Kp_Ankle", "Kd_Ankle", "Kp_Knee", "Kd_Knee", "Cost"]
        data       = [r[:5] for r in grid_results]
        df_grid    = pd.DataFrame(data, columns=cols)
        valid_grid = df_grid[df_grid['Cost'] < 1e8].sort_values('Cost')

        if valid_grid.empty:
            print("\n[ERRORE] Tutte le simulazioni di Phase 1 sono fallite.")
            print("Decommenta il TEST SINGOLA RUN per diagnosticare.")
            exit(1)

        best_phase1 = valid_grid.iloc[0].to_dict()

        # ------------------------------------------------------------------
        # PHASE 2 — Differential Evolution
        # support_mapping non viene passato: i worker lo caricano da JSON
        # ------------------------------------------------------------------
        best_phase2 = phase2_differential_evolution(best_phase1, num_cores)


        # ------------------------------------------------------------------
        # RISULTATI FINALI
        # ------------------------------------------------------------------
        save_and_print_results(grid_results, best_de=best_phase2)

        print("\n" + "="*60)
        print("OTTIMIZZAZIONE COMPLETATA")
        print("="*60)
        print(f"\nParametri finali consigliati:")
        print(f"  SEA_Ankle -> Kp = {best_phase2['Kp_Ankle']}  |  Kd = {best_phase2['Kd_Ankle']}")
        print(f"  SEA_Knee  -> Kp = {best_phase2['Kp_Knee']}   |  Kd = {best_phase2['Kd_Knee']}")
        print(f"  Costo minimo: {best_phase2['Cost']:.4f}")

    except KeyboardInterrupt:
        print("\n[STOP] Interrotto dall'utente.")
        subprocess.call("taskkill /F /IM cmc.exe", shell=True)
        exit(0)
