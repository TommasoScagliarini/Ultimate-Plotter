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
"""

import os
import shutil
import subprocess
import multiprocessing as mp
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import itertools
import time
import signal
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

# Directory di lavoro SENZA SPAZI: CMC spezza i path con spazi come fossero liste
WORK_DIR    = r"C:\CMC_Sweep"
RESULTS_DIR = os.path.join(WORK_DIR, "sweep_results")
SUMMARY_CSV = os.path.join(WORK_DIR, "sweep_summary_global.csv")

TARGET_ACTUATORS = ["SEA_Ankle", "SEA_Knee"]

# Mapping: colonna nel file _Kinematics_q.sto simulato -> colonna nel riferimento sano
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
    assoluti con spazi (che puntano a file esistenti su disco) con copie
    degli stessi file in WORK_DIR, aggiornando il testo del tag.
    Salva il file XML modificato su se stesso.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        modified = False

        for elem in root.iter():
            if not elem.text or not elem.text.strip():
                continue
            text = elem.text.strip()
            # Cerca path assoluti con spazi che puntano a file esistenti
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
    1. Legge il setup CMC originale e copia in WORK_DIR tutti i file
       referenziati in SETUP_FILE_TAGS.
    2. Copia anche il modello base.
    3. Patcha i path interni di ogni XML copiato (es. GRF dentro
       Externall_Loads.xml) per eliminare spazi residui.
    Restituisce un dict {path_assoluto_originale: path_in_workdir}.
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
                print(f"  [WARN] File di supporto non trovato: {src}  (tag: <{tag}>)")

    except Exception as e:
        print(f"  [WARN] copy_support_files_to_workdir: {e}")

    # Copia anche il modello base
    if os.path.exists(MODEL_FILE_BASE):
        dst_model = os.path.join(WORK_DIR, os.path.basename(MODEL_FILE_BASE))
        shutil.copy2(MODEL_FILE_BASE, dst_model)
        mapping[MODEL_FILE_BASE] = dst_model
        print(f"  [COPY] {os.path.basename(MODEL_FILE_BASE)}")

    # Patcha i path interni di ogni XML copiato in WORK_DIR
    print("  Patching path interni degli XML...")
    for fname in os.listdir(WORK_DIR):
        if fname.lower().endswith('.xml'):
            _patch_xml_internal_paths(os.path.join(WORK_DIR, fname))

    return mapping

def _read_sto(filepath):
    """
    Legge un file .sto/.mot di OpenSim saltando l'header fino a 'endheader'.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip() == 'endheader':
            start = i + 1
            break
    df = pd.read_csv(filepath, sep='\t', skiprows=start)
    df.columns = df.columns.str.strip()
    return df
# ==============================================================================
# 3. COST FUNCTION
# ==============================================================================

def evaluate_run_cost(results_dir):
    """
    Cost = W_KIN  * sum((q_sim - q_ref)^2)
         + W_CHAT * sum(diff(omega_motor)^2)
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

            # I nomi delle colonne nel file states seguono il pattern:
            # /forceset/SEA_Ankle/motor_speed  oppure  SEA_Ankle.motor_speed
            # Cerca qualsiasi colonna che contenga 'motor_speed'
            speed_cols = [c for c in df_states.columns if 'motor_speed' in c.lower()]

            if not speed_cols:
                print(f"  [WARN] Nessuna colonna 'motor_speed' trovata in _states.sto")
                print(f"         Colonne disponibili: {list(df_states.columns)}")
            else:
                for col in speed_cols:
                    chatter_cost += float(
                        np.sum(np.diff(df_states[col].values) ** 2)
                    )

        # ── Pesi ──────────────────────────────────────────────────────────────
        W_KIN, W_CHAT = 1.0, 1.0
        return W_KIN * kin_cost + W_CHAT * chatter_cost

    except Exception as e:
        print(f"  [WARN] evaluate_run_cost: {e}")
        return 1e9


# ==============================================================================
# 4. MODIFICA DEL MODELLO E DEL SETUP
# ==============================================================================

def _set_actuator_param(root, actuator_name, param_name, value):
    """
    Trova <SeriesElasticActuator name="..."> e imposta il tag <param_name>.
    Restituisce True se riuscito.
    """
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
    """
    Sostituisce i path dei file di supporto nel setup XML con i corrispondenti
    path in WORK_DIR (senza spazi), usando il mapping costruito da
    copy_support_files_to_workdir().
    """
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
    """
    Crea temp_model.osim e temp_setup.xml nella run_dir.
    Restituisce (temp_model_path, temp_setup_path) oppure (None, None).
    """
    os.makedirs(run_dir, exist_ok=True)
    temp_model = os.path.join(run_dir, "temp_model.osim")
    temp_setup = os.path.join(run_dir, "temp_setup.xml")

    # ── Modello: parte dalla copia in WORK_DIR (senza spazi) ─────────────────
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

    # ── Setup CMC ─────────────────────────────────────────────────────────────
    try:
        tree_setup = ET.parse(SETUP_FILE_BASE)
    except ET.ParseError as e:
        print(f"  [ERROR] Parsing setup: {e}")
        return None, None

    root_setup = tree_setup.getroot()
    cmc_tool   = root_setup.find('.//CMCTool')

    if cmc_tool is not None:
        # 1. Sostituisce i path con le copie in WORK_DIR (nessuno spazio)
        _resolve_setup_paths(cmc_tool, support_mapping)

        # 2. Sovrascrivi i tre tag specifici dello sweep
        for tag, value in [('model_file',        temp_model),
                           ('results_directory', run_dir),
                           ('name',              run_name)]:
            node = cmc_tool.find(f'.//{tag}')
            if node is not None:
                node.text = value

    tree_setup.write(temp_setup, encoding='unicode', xml_declaration=False)

    return temp_model, temp_setup

# ==============================================================================
# SISTEMA DI PROGRESS (aggiungilo prima delle funzioni di phase)
# ==============================================================================

def _progress_listener(queue, total_runs, phase_label):
    """
    Gira nel processo main in un thread separato.
    Ascolta la queue, aggiorna la barra e calcola l'ETA.
    Manda 'DONE' per terminare.
    """
    import threading

    completed  = 0
    times      = []   # tempi di ogni run completata
    start_glob = time.time()

    def _bar(n, tot, eta_str):
        pct    = n / tot
        filled = int(pct * 30)
        bar    = '#' * filled + '-' * (30 - filled)
        print(f"\r  {phase_label}  [{bar}] {n}/{tot}  {pct*100:5.1f}%  ETA: {eta_str}   ",
              end='', flush=True)

    _bar(0, total_runs, '--:--')

    while True:
        item = queue.get()
        if item == 'DONE':
            print()   # newline finale
            break

        elapsed_run = item   # il worker manda il tempo impiegato
        completed  += 1
        times.append(elapsed_run)

        avg_time = sum(times) / len(times)
        remaining = (total_runs - completed) * avg_time
        eta_str   = time.strftime('%M:%S', time.gmtime(remaining))

        _bar(completed, total_runs, eta_str)

    total_elapsed = time.time() - start_glob
    print(f"  Completato in {time.strftime('%H:%M:%S', time.gmtime(total_elapsed))}")


def _make_progress_pool(total_runs, phase_label, num_cores):
    """
    Crea la Queue e avvia il listener in un thread daemon.
    Restituisce (queue, thread) — chiudi con queue.put('DONE').
    """
    import threading
    q = mp.Manager().Queue()
    t = threading.Thread(
        target=_progress_listener,
        args=(q, total_runs, phase_label),
        daemon=True
    )
    t.start()
    return q, t

# ==============================================================================
# 5. WORKER
# ==============================================================================

def run_cmc_worker(params):
    """
    params: (kp_ankle, kd_ankle, kp_knee, kd_knee, run_id, support_mapping)
    Restituisce: (kp_ankle, kd_ankle, kp_knee, kd_knee, cost)
    """
    kp_ankle, kd_ankle, kp_knee, kd_knee, run_id, support_mapping = params[:6]
    progress_queue = params[6] if len(params) > 6 else None

    run_name = (f"run{run_id:04d}"
                f"_KpK{kp_knee}_KdK{kd_knee}"
                f"_KpA{kp_ankle}_KdA{kd_ankle}")
    run_dir  = os.path.join(RESULTS_DIR, run_name)

    # Salta se gia' eseguita (utile in caso di restart)
    existing = [f for f in os.listdir(run_dir)
                if f.endswith('_controls.sto')] \
               if os.path.isdir(run_dir) else []
    if existing:
        cost = evaluate_run_cost(run_dir)
        print(f"  [SKIP] {run_name} -> costo: {cost:.4f} (gia' presente)")
        return (kp_ankle, kd_ankle, kp_knee, kd_knee, cost)

    temp_model, temp_setup = build_run_files(
        kp_ankle, kd_ankle, kp_knee, kd_knee,
        run_dir, run_name, support_mapping
    )
    if temp_model is None:
        return (kp_ankle, kd_ankle, kp_knee, kd_knee, 1e9)

    cmd = [CMC_EXE, "-Setup", temp_setup, "-Library", PLUGIN_DLL]

    t0 = time.time()
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=WORK_DIR,
            check=True,
            timeout=1800          # 30 minuti max per run
        )
        cost = evaluate_run_cost(run_dir)

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {run_name}")
        cost = 1e9

    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode('utf-8', errors='replace') if e.stderr else ""
        stdout_text = e.stdout.decode('utf-8', errors='replace') if e.stdout else ""
        err_lines = [l for l in stderr_text.splitlines() if l.strip()][-20:]
        out_lines = [l for l in stdout_text.splitlines() if l.strip()][-20:]
        print(f"  [FAIL] {run_name} — returncode {e.returncode}")
        if err_lines:
            print("  [STDERR]\n    " + "\n    ".join(err_lines))
        if out_lines:
            print("  [STDOUT]\n    " + "\n    ".join(out_lines))
        cost = 1e9

    elapsed = time.time() - t0
    print(f"  [{run_id:04d}] Kp Knee:{kp_knee:5d} Kd Knee:{kd_knee:4d} | "
          f"Kp Ankle:{kp_ankle:5d} Kd Ankle:{kd_ankle:4d} -> "
          f"costo: {cost:.4f}  ({elapsed:.1f}s)")

    if progress_queue is not None:
        progress_queue.put(elapsed)   # manda il tempo al listener

    return (kp_ankle, kd_ankle, kp_knee, kd_knee, cost)


# ==============================================================================
# 6. PHASE 1 — GRID SEARCH GROSSOLANO
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

    print(f"Combinazioni totali: {len(tasks)}")
    print(f"Core utilizzati:     {num_cores}\n")

    q, t = _make_progress_pool(len(tasks), "Phase 1", num_cores)

    # Aggiungi la queue a ogni task
    tasks = [(*combo, run_id, support_mapping, q)
             for run_id, combo in enumerate(combinations, start=1)]

    with mp.Pool(num_cores) as pool:
        results = pool.map(run_cmc_worker, tasks)

    q.put('DONE')
    t.join()

    return results


# ==============================================================================
# 7. PHASE 2 — DIFFERENTIAL EVOLUTION (raffinamento)
# ==============================================================================

_de_counter        = mp.Value('i', 10000)
_support_mapping_g = {}


def _de_cost_wrapper(x):
    kp_ankle = int(round(x[0]))
    kd_ankle = int(round(x[1]))
    kp_knee  = int(round(x[2]))
    kd_knee  = int(round(x[3]))

    with _de_counter.get_lock():
        _de_counter.value += 1
        run_id = _de_counter.value

    result = run_cmc_worker(
        (kp_ankle, kd_ankle, kp_knee, kd_knee, run_id, _support_mapping_g)
    )
    return result[-1]


def phase2_differential_evolution(best_from_phase1, num_cores, support_mapping):
    global _support_mapping_g
    _support_mapping_g = support_mapping

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
        maxiter=15,
        popsize=6,
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
# 8. ANALISI E SALVATAGGIO RISULTATI
# ==============================================================================

def save_and_print_results(all_results, best_de=None):
    cols = ["Kp_Ankle", "Kd_Ankle", "Kp_Knee", "Kd_Knee", "Cost"]
    df = pd.DataFrame(all_results, columns=cols)
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

        improvement = (best_grid['Cost'] - best_de['Cost']) / best_grid['Cost'] * 100
        print(f"\n  Miglioramento rispetto al grid: {improvement:.1f}%")

    return best_grid.to_dict()


# ==============================================================================
# 9. MAIN
# ==============================================================================

if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal.SIG_DFL)  # ripristina Ctrl+C

    os.makedirs(WORK_DIR,    exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    num_cores = max(1, mp.cpu_count() // 4)  # conservativo: evita contesa risorse

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

    # Copia i file di supporto in WORK_DIR e patcha i path interni
    print("Copio i file di supporto in WORK_DIR...")
    support_mapping = copy_support_files_to_workdir()
    print(f"  {len(support_mapping)} file copiati.\n")

    # ------------------------------------------------------------------
    # TEST SINGOLA RUN — decommenta per diagnosticare prima del sweep
    # ------------------------------------------------------------------
    print("\n--- TEST SINGOLA RUN ---")
    print("  Avvio simulazione...", flush=True)
    t0 = time.time()
    result = run_cmc_worker((7500, 20, 2500, 20, 9999, support_mapping, None))
    elapsed = time.time() - t0
    print(f"  Completata in {elapsed:.1f}s")
    print(f"  Costo:        {result[-1]:.4f}")
    print(f"  kin_cost e chatter_cost sono stampati dentro evaluate_run_cost")
    exit(0)

    try:
        # ------------------------------------------------------------------
        # PHASE 1 — Grid search
        # ------------------------------------------------------------------
        grid_results = phase1_grid_search(num_cores, support_mapping)

        cols       = ["Kp_Ankle", "Kd_Ankle", "Kp_Knee", "Kd_Knee", "Cost"]
        df_grid    = pd.DataFrame(grid_results, columns=cols)
        valid_grid = df_grid[df_grid['Cost'] < 1e8].sort_values('Cost')

        if valid_grid.empty:
            print("\n[ERRORE] Tutte le simulazioni di Phase 1 sono fallite.")
            print("Decommenta il TEST SINGOLA RUN per diagnosticare.")
            exit(1)

        best_phase1 = valid_grid.iloc[0].to_dict()

        # ------------------------------------------------------------------
        # PHASE 2 — Differential Evolution
        # ------------------------------------------------------------------
        best_phase2 = phase2_differential_evolution(best_phase1, num_cores,
                                                    support_mapping)

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