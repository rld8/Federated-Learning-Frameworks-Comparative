#!/usr/bin/env bash
# Ejecuta la matriz completa de experimentos Flower del TFG.
# Uso:
#   ./run_flower_experiments.sh 1   # modo secuencial: 1 ClientApp activo por GPU
#   ./run_flower_experiments.sh 2   # modo concurrente: varios ClientApp comparten la GPU
#
# Variables opcionales:
#   REPEATS=3
#   INCLUDE_BASELINE=1
#   EXPERIMENT_FILTER=e1
#   COOLDOWN_SECONDS=5
#   LR=0.02
#   BATCH_SIZE=32
#   FLWR_LOG_LEVEL=DEBUG
#   TFG_VERBOSE_CLIENT=1

set -uo pipefail

MODE="${1:-}"
if [[ "$MODE" != "1" && "$MODE" != "2" ]]; then
  echo "Uso: $0 <modo>"
  echo "  modo 1: una tarea de cliente activa cada vez"
  echo "  modo 2: clientes de la ronda concurrentes"
  exit 1
fi

PROJECT_DIR="${FLOWER_PROJECT_DIR:-$HOME/Escritorio/tfg/FLOWER}"
REPEATS="${REPEATS:-3}"
INCLUDE_BASELINE="${INCLUDE_BASELINE:-1}"
STRATEGY="${STRATEGY:-fedavg}"
EXPERIMENT_FILTER="${EXPERIMENT_FILTER:-}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-5}"
LR="${LR:-0.02}"
BATCH_SIZE="${BATCH_SIZE:-32}"
FLOWER_LOG_LEVEL="${FLWR_LOG_LEVEL:-DEBUG}"
TFG_VERBOSE_CLIENT="${TFG_VERBOSE_CLIENT:-1}"
MODE_NAME=$([[ "$MODE" == "1" ]] && echo "sequential" || echo "concurrent")
STAMP="$(date +%Y%m%d_%H%M%S)"
SUITE_DIR="$PROJECT_DIR/resultados/experiment_suite/$MODE_NAME/$STAMP"
SUMMARY_CSV="$SUITE_DIR/summary.csv"

# label|clientes_totales|clientes_por_ronda|rondas|epocas_locales
EXPERIMENTS=(
  "e0_baseline|10|10|30|2"
  "e1_principal|5|5|20|10"
  "e2_scalability_c4|4|4|10|2"
  "e2_scalability_c5|5|5|10|2"
  "e2_scalability_c8|8|8|10|2"
  "e3_flexibility_frac05|8|4|20|10"
  "e3_flex_mini|8|4|3|1"
  "e3_flex_tiny|4|2|2|1"
)

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "ERROR: no existe PROJECT_DIR=$PROJECT_DIR"
  exit 1
fi
cd "$PROJECT_DIR"

mkdir -p "$SUITE_DIR"

echo 'framework,mode,experiment,seed,clients,clients_per_round,rounds,local_epochs,batch_size,learning_rate,status,exit_code,duration_seconds,client_updates,estimated_train_examples,estimated_examples_per_second,final_accuracy,final_loss,final_train_accuracy,final_train_loss,gpu_mem_max_mib,gpu_mem_mean_mib,gpu_util_max_pct,gpu_util_mean_pct,power_max_w,power_mean_w,estimated_energy_wh,temp_max_c,max_rss_kb,run_dir' > "$SUMMARY_CSV"

for cmd in flwr python nvidia-smi /usr/bin/time; do
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "ERROR: falta el comando $cmd"
    exit 1
  }
done

if ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi no funciona. No ejecutes experimentos porque las métricas GPU no serán válidas."
  nvidia-smi
  exit 1
fi

if ! flwr run --help 2>&1 | grep -q -- '--federation-config'; then
  echo "ERROR: tu versión de Flower no soporta --federation-config por ejecución."
  echo "Actualiza Flower o configura manualmente la federación local."
  exit 1
fi

if command -v ray >/dev/null 2>&1; then
  echo "Comprobando que Ray detecta la GPU..."
  if ! python - <<'PY'
import sys
try:
    import ray
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    resources = ray.cluster_resources()
    print(resources)
    gpu_count = float(resources.get("GPU", 0.0))
    ray.shutdown()
    if gpu_count < 1.0:
        sys.exit(2)
except Exception as exc:
    print(f"ERROR comprobando Ray/GPU: {exc}")
    sys.exit(3)
PY
  then
    echo "ERROR: Ray no detecta GPU. Revisa nvidia-smi, drivers y el entorno conda."
    exit 1
  fi
  ray stop --force >/dev/null 2>&1 || true
else
  echo "AVISO: no encuentro el comando ray. Flower puede seguir funcionando, pero no podré limpiar Ray con ray stop."
fi

cat > "$SUITE_DIR/sitecustomize.py" <<'PY'
import os
import random

seed = int(os.environ.get("TFG_SEED", "42"))
random.seed(seed)

try:
    import numpy as np
    np.random.seed(seed)
except Exception:
    pass

try:
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
except Exception:
    pass
PY

snapshot_environment() {
  local dir="$1"
  mkdir -p "$dir/environment" "$dir/config" "$dir/code"
  date --iso-8601=seconds > "$dir/environment/start_time.txt"
  uname -a > "$dir/environment/uname.txt" 2>&1 || true
  lscpu > "$dir/environment/lscpu.txt" 2>&1 || true
  free -h > "$dir/environment/free_h.txt" 2>&1 || true
  nvidia-smi -q > "$dir/environment/nvidia_smi_q.txt" 2>&1 || true
  nvidia-smi > "$dir/environment/nvidia_smi.txt" 2>&1 || true
  python --version > "$dir/environment/python_version.txt" 2>&1 || true
  pip freeze > "$dir/environment/pip_freeze.txt" 2>&1 || true
  conda env export > "$dir/environment/conda_env.yml" 2>&1 || true
  git rev-parse HEAD > "$dir/environment/git_commit.txt" 2>&1 || true
  git status --short > "$dir/environment/git_status.txt" 2>&1 || true
  cp -a pyproject.toml "$dir/code/" 2>/dev/null || true
  cp -a pytorchexample "$dir/code/" 2>/dev/null || true
}

MONITOR_PIDS=()

start_monitors() {
  local dir="$1"
  mkdir -p "$dir"

  nvidia-smi --query-gpu=timestamp,index,name,pstate,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem --format=csv -l 1 > "$dir/gpu.csv" 2>&1 &
  MONITOR_PIDS+=("$!")

  nvidia-smi dmon -s pucvmet -d 1 > "$dir/gpu_dmon.txt" 2>&1 &
  MONITOR_PIDS+=("$!")

  nvidia-smi pmon -s um -d 1 > "$dir/gpu_pmon.txt" 2>&1 &
  MONITOR_PIDS+=("$!")

  vmstat -t 1 > "$dir/vmstat.txt" 2>&1 &
  MONITOR_PIDS+=("$!")

  (
    while true; do
      echo "===== $(date --iso-8601=seconds) ====="
      ps -eo pid,ppid,%cpu,%mem,rss,vsz,etimes,cmd --sort=-rss | head -80
      sleep 1
    done
  ) > "$dir/process_snapshots.txt" 2>&1 &
  MONITOR_PIDS+=("$!")
}

stop_monitors() {
  local pid
  for pid in "${MONITOR_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait "${MONITOR_PIDS[@]:-}" 2>/dev/null || true
  MONITOR_PIDS=()
}

extract_metrics() {
  local run_dir="$1" label="$2" seed="$3" clients="$4" cpr="$5" rounds="$6" epochs="$7" status="$8" exit_code="$9" duration="${10}"

  python - "$run_dir" "$SUMMARY_CSV" "$label" "$seed" "$clients" "$cpr" "$rounds" "$epochs" "$LR" "$BATCH_SIZE" "$MODE_NAME" "$status" "$exit_code" "$duration" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

run_dir, summary, label, seed, clients, cpr, rounds, epochs, lr, bs, mode, status, exit_code, duration = sys.argv[1:]
run = Path(run_dir)
text = (run / "stdout.log").read_text(errors="ignore") if (run / "stdout.log").exists() else ""

def vals(pattern):
    return [float(x) for x in re.findall(pattern, text, flags=re.I)]

accs = vals(r"Server-side evaluation round\s+\d+\s*:\s*loss=[0-9.eE+-]+,\s*accuracy=([0-9.eE+-]+)")
losses = vals(r"Server-side evaluation round\s+\d+\s*:\s*loss=([0-9.eE+-]+),\s*accuracy=[0-9.eE+-]+")
train_losses = vals(r"train_loss['\"=: ]+([0-9.eE+-]+)")

with (run / "metrics_extracted.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["round", "accuracy", "loss"])
    for rnd, loss, acc in re.findall(r"Server-side evaluation round\s+(\d+)\s*:\s*loss=([0-9.eE+-]+),\s*accuracy=([0-9.eE+-]+)", text, flags=re.I):
        w.writerow([rnd, acc, loss])

def num(s):
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", str(s))
    return float(m.group()) if m else None

gpu_rows = []
gpu_file = run / "gpu.csv"
if gpu_file.exists():
    try:
        with gpu_file.open(errors="ignore") as f:
            for row in csv.DictReader(f):
                gpu_rows.append(row)
    except Exception:
        pass

def col_values(fragment):
    out = []
    for row in gpu_rows:
        for k, v in row.items():
            if fragment.lower() in k.lower():
                x = num(v)
                if x is not None:
                    out.append(x)
                break
    return out

mem = col_values("memory.used")
util = col_values("utilization.gpu")
power = col_values("power.draw")
temp = col_values("temperature.gpu")

time_text = (run / "time.txt").read_text(errors="ignore") if (run / "time.txt").exists() else ""
m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", time_text)
max_rss = int(m.group(1)) if m else None

clients_i = int(clients)
cpr_i = int(cpr)
rounds_i = int(rounds)
epochs_i = int(epochs)
dur = float(duration)
updates = cpr_i * rounds_i
estimated_examples = int(40000 * (cpr_i / clients_i) * epochs_i * rounds_i)
energy = (sum(power) / len(power)) * dur / 3600 if power and dur > 0 else None

row = {
    "framework": "flower",
    "mode": mode,
    "experiment": label,
    "seed": int(seed),
    "clients": clients_i,
    "clients_per_round": cpr_i,
    "rounds": rounds_i,
    "local_epochs": epochs_i,
    "batch_size": int(bs),
    "learning_rate": float(lr),
    "status": status,
    "exit_code": int(exit_code),
    "duration_seconds": dur,
    "client_updates": updates,
    "estimated_train_examples": estimated_examples,
    "estimated_examples_per_second": estimated_examples / dur if dur > 0 else None,
    "final_accuracy": accs[-1] if accs else None,
    "final_loss": losses[-1] if losses else None,
    "final_train_accuracy": None,
    "final_train_loss": train_losses[-1] if train_losses else None,
    "gpu_mem_max_mib": max(mem) if mem else None,
    "gpu_mem_mean_mib": sum(mem) / len(mem) if mem else None,
    "gpu_util_max_pct": max(util) if util else None,
    "gpu_util_mean_pct": sum(util) / len(util) if util else None,
    "power_max_w": max(power) if power else None,
    "power_mean_w": sum(power) / len(power) if power else None,
    "estimated_energy_wh": energy,
    "temp_max_c": max(temp) if temp else None,
    "max_rss_kb": max_rss,
    "run_dir": str(run.resolve()),
}

(run / "summary.json").write_text(json.dumps(row, indent=2))
fields = list(row.keys())
with open(summary, "a", newline="") as f:
    csv.DictWriter(f, fieldnames=fields).writerow(row)
PY
}

if [[ "$MODE" == "1" ]]; then
  echo "Modo 1: Flower ejecutará un único ClientApp activo por GPU. Los ClientApp de simulación son efímeros."
else
  echo "Modo 2: Flower intentará ejecutar concurrentemente todos los clientes de cada ronda. Puede producir OOM."
fi

echo "Suite: $SUITE_DIR"

for spec in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r label clients cpr rounds epochs <<< "$spec"

  [[ "$label" == "e0_baseline" && "$INCLUDE_BASELINE" != "1" ]] && continue
  [[ -n "$EXPERIMENT_FILTER" && "$label" != *"$EXPERIMENT_FILTER"* ]] && continue

  for ((rep=0; rep<REPEATS; rep++)); do
    seed=$((42 + rep))

    fraction=$(python - <<PY
print($cpr / $clients)
PY
)

    if [[ "$MODE" == "1" ]]; then
  gpu_share="1.0"
else
  gpu_share="0.5"
fi

    run_name="flower_${label}_${MODE_NAME}_clients${clients}_cpr${cpr}_rounds${rounds}_ep${epochs}_lr002_bs32_seed${seed}"
    run_dir="$SUITE_DIR/$run_name"
    mkdir -p "$run_dir/config" "$run_dir/environment" "$run_dir/code"

    snapshot_environment "$run_dir"

    cat > "$run_dir/config/run_config.txt" <<CFG
num-server-rounds=$rounds
fraction-train=$fraction
fraction-evaluate=0.0
local-epochs=$epochs
learning-rate=$LR
batch-size=$BATCH_SIZE
seed=$seed
strategy=$STRATEGY
CFG

    cat > "$run_dir/config/federation_config.txt" <<CFG
num-supernodes=$clients
client-resources-num-cpus=1
client-resources-num-gpus=$gpu_share
verbose=true
init-args-logging-level=$FLOWER_LOG_LEVEL
init-args-log-to-driver=true
CFG

    cat > "$run_dir/live_commands.txt" <<CFG
# Ver log principal del experimento
tail -f "$run_dir/stdout.log"

# Ver consumo GPU en directo
watch -n 1 nvidia-smi

# Ver CSV de GPU
tail -f "$run_dir/gpu.csv"

# Ver procesos más pesados
watch -n 1 'ps -eo pid,ppid,%cpu,%mem,rss,vsz,etimes,cmd --sort=-rss | head -30'

# Ver logs de Flower/SuperLink
tail -f ~/.flwr/local-superlink/superlink.log
CFG

    echo "===== $run_name ====="
    echo "Run dir: $run_dir"
    echo "Logs útiles: cat $run_dir/live_commands.txt"

    rm -f final_model.pt
    command -v ray >/dev/null 2>&1 && ray stop --force >/dev/null 2>&1 || true

    start_monitors "$run_dir"
    start_epoch=$(date +%s)

    set +e
    RUN_CONFIG="num-server-rounds=$rounds fraction-train=$fraction fraction-evaluate=0.0 local-epochs=$epochs learning-rate=$LR batch-size=$BATCH_SIZE strategy=\"$STRATEGY\""
    echo "RUN_CONFIG=$RUN_CONFIG" | tee -a "$run_dir/stdout.log"

    PYTHONPATH="$SUITE_DIR:${PYTHONPATH:-}" \
    TFG_SEED="$seed" \
    PYTHONHASHSEED="$seed" \
    PYTHONUNBUFFERED=1 \
    FLWR_LOG_LEVEL="$FLOWER_LOG_LEVEL" \
    RAY_DEDUP_LOGS=0 \
    TFG_VERBOSE_CLIENT="$TFG_VERBOSE_CLIENT" \
      /usr/bin/time -v -o "$run_dir/time.txt" \
      flwr run . --stream \
      --run-config="$RUN_CONFIG" \
--federation-config="num-supernodes=$clients init-args-num-cpus=2 client-resources-num-cpus=1 client-resources-num-gpus=$gpu_share" \
      2>&1 | tee -a "$run_dir/stdout.log"
    exit_code=${PIPESTATUS[0]}

    # Flower/Ray a veces devuelven exit_code 0 aunque los clientes hayan fallado.
    # Por eso también inspeccionamos el log de la ejecución.
    if grep -qiE "Simulation Engine crashed|Exit Code: 700|An unhandled exception occurred|ActorPool is empty|Traceback|ClientAppException|RayTaskError|An exception was raised when processing a message|Received [0-9]+ results and [1-9][0-9]* failures|Received 0 results|CUDA out of memory|out of memory|OOM" "$run_dir/stdout.log"; then
      exit_code=700
    fi
    set -u

    end_epoch=$(date +%s)
    duration=$((end_epoch - start_epoch))

    stop_monitors
    command -v ray >/dev/null 2>&1 && ray stop --force >/dev/null 2>&1 || true

    if grep -qi "Simulation Engine crashed\|Exit Code: 700\|An unhandled exception occurred\|Simulation raised an exception\|ActorPool is empty\|Traceback" "$run_dir/stdout.log"; then
      exit_code=700
    fi

    grep -Eo "Successfully started run [0-9]+" "$run_dir/stdout.log" | awk '{print $4}' | tail -1 > "$run_dir/run_id.txt" 2>/dev/null || true

    [[ -f final_model.pt ]] && cp final_model.pt "$run_dir/final_model.pt"

    status=$([[ "$exit_code" == "0" ]] && echo success || echo failed)
    extract_metrics "$run_dir" "$label" "$seed" "$clients" "$cpr" "$rounds" "$epochs" "$status" "$exit_code" "$duration"
    date --iso-8601=seconds > "$run_dir/environment/end_time.txt"

    echo "$run_name -> $status (exit=$exit_code, ${duration}s)"
    echo "Resumen individual: $run_dir/summary.json"

    sleep "$COOLDOWN_SECONDS"
  done
done

echo "Finalizado. Resumen: $SUMMARY_CSV"
