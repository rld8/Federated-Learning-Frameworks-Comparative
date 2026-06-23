#!/usr/bin/env bash
# Ejecuta la matriz completa de experimentos NVIDIA FLARE del TFG.
# Uso: ./run_nvflare_experiments.sh 1   # un hilo/tarea de cliente activa cada vez
#      ./run_nvflare_experiments.sh 2   # tareas de cliente concurrentes
# Variables opcionales: REPEATS=3 INCLUDE_BASELINE=1 EXPERIMENT_FILTER=e1 COOLDOWN_SECONDS=5
# BASE_JOB_DIR puede cambiarse si tu job base tiene otro nombre.

set -uo pipefail

MODE="${1:-}"
if [[ "$MODE" != "1" && "$MODE" != "2" ]]; then
  echo "Uso: $0 <modo>"
  echo "  modo 1: un hilo/tarea de cliente activo cada vez"
  echo "  modo 2: clientes de la ronda concurrentes"
  exit 1
fi

PROJECT_DIR="${NVFLARE_PROJECT_DIR:-$HOME/Escritorio/tfg/NFLARE}"
BASE_JOB_DIR="${BASE_JOB_DIR:-$PROJECT_DIR/jobs/nvflare_resnet18_cifar10_clients10_frac10_rounds30_ep2_lr002_bs32_no_persist}"
REPEATS="${REPEATS:-3}"
INCLUDE_BASELINE="${INCLUDE_BASELINE:-1}"
EXPERIMENT_FILTER="${EXPERIMENT_FILTER:-}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-5}"
LR="${LR:-0.02}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TFG_FL_STRATEGY="${TFG_FL_STRATEGY:-fedavg}"
TFG_FEDPROX_MU="${TFG_FEDPROX_MU:-0.001}"
MODE_NAME=$([[ "$MODE" == "1" ]] && echo "sequential" || echo "concurrent")
STAMP="$(date +%Y%m%d_%H%M%S)"
SUITE_DIR="$PROJECT_DIR/resultados/experiment_suite/$MODE_NAME/$STAMP"
SUMMARY_CSV="$SUITE_DIR/summary.csv"

# label|clientes_totales|clientes_por_ronda|min_clients|rondas|epocas_locales
EXPERIMENTS=(
  "e0_baseline|10|10|10|30|2"
  "e1_principal|5|5|5|20|10"
  "e2_scalability_c4|4|4|4|10|2"
  "e2_scalability_c5|5|5|5|10|2"
  "e2_scalability_c8|8|8|8|10|2"
  "e3_flexibility_frac05|8|4|4|20|10"
  "e3_flex_mini|8|4|4|3|1"
  "e3_flex_tiny|4|2|2|2|1"
)

mkdir -p "$SUITE_DIR"
echo 'framework,mode,experiment,seed,clients,clients_per_round,rounds,local_epochs,batch_size,learning_rate,status,exit_code,duration_seconds,client_updates,estimated_train_examples,estimated_examples_per_second,final_accuracy,final_loss,final_train_accuracy,final_train_loss,gpu_mem_max_mib,gpu_mem_mean_mib,gpu_util_max_pct,gpu_util_mean_pct,power_max_w,power_mean_w,estimated_energy_wh,temp_max_c,max_rss_kb,run_dir' > "$SUMMARY_CSV"

if [[ ! -d "$PROJECT_DIR" ]]; then echo "ERROR: no existe PROJECT_DIR=$PROJECT_DIR"; exit 1; fi
if [[ ! -d "$BASE_JOB_DIR" ]]; then echo "ERROR: no existe BASE_JOB_DIR=$BASE_JOB_DIR"; exit 1; fi
cd "$PROJECT_DIR"
for cmd in nvflare python nvidia-smi /usr/bin/time; do command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: falta $cmd"; exit 1; }; done

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
  cp -a "$PROJECT_DIR/code"/*.py "$dir/code/" 2>/dev/null || true
}

MONITOR_PIDS=()
start_monitors() {
  local dir="$1"
  nvidia-smi --query-gpu=timestamp,index,name,pstate,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem --format=csv -l 1 > "$dir/gpu.csv" 2>&1 & MONITOR_PIDS+=("$!")
  nvidia-smi dmon -s pucvmet -d 1 > "$dir/gpu_dmon.txt" 2>&1 & MONITOR_PIDS+=("$!")
  nvidia-smi pmon -s um -d 1 > "$dir/gpu_pmon.txt" 2>&1 & MONITOR_PIDS+=("$!")
  vmstat -t 1 > "$dir/vmstat.txt" 2>&1 & MONITOR_PIDS+=("$!")
  (while true; do echo "===== $(date --iso-8601=seconds) ====="; ps -eo pid,ppid,%cpu,%mem,rss,vsz,etimes,cmd --sort=-rss | head -100; sleep 1; done) > "$dir/process_snapshots.txt" 2>&1 & MONITOR_PIDS+=("$!")
}
stop_monitors() { local pid; for pid in "${MONITOR_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done; wait "${MONITOR_PIDS[@]:-}" 2>/dev/null || true; MONITOR_PIDS=(); }

patch_job() {
  local job_dir="$1" clients="$2" min_clients="$3" rounds="$4" epochs="$5" seed="$6"
  python - "$job_dir" "$clients" "$min_clients" "$rounds" "$epochs" "$BATCH_SIZE" "$LR" "$seed" <<'PY'
import re, sys
from pathlib import Path
job=Path(sys.argv[1]); clients, min_clients, rounds, epochs, bs, lr, seed=sys.argv[2:]
server=list(job.rglob("config_fed_server.conf"))
client=list(job.rglob("config_fed_client.conf"))
if not server or not client:
    raise SystemExit("No se encontraron config_fed_server.conf/config_fed_client.conf en el job base")
report=[]
def patch_assignment(text, key, value):
    patterns=[rf'(?m)(\b{re.escape(key)}\s*[:=]\s*)\d+', rf'(?m)(["\']{re.escape(key)}["\']\s*:\s*)\d+']
    total=0
    for p in patterns:
        text,n=re.subn(p,rf'\g<1>{value}',text); total+=n
    return text,total
for path in server:
    text=path.read_text()
    text,n1=patch_assignment(text,"num_rounds",rounds)
    text,n2=patch_assignment(text,"min_clients",min_clients)
    path.write_text(text); report.append(f"{path}: num_rounds={n1}, min_clients={n2}")
for path in client:
    text=path.read_text()
    replacements={
      r'--num_clients\s+\d+':f'--num_clients {clients}',
      r'--batch_size\s+\d+':f'--batch_size {bs}',
      r'--local_epochs\s+\d+':f'--local_epochs {epochs}',
      r'--lr\s+[0-9.eE+-]+':f'--lr {lr}',
      r"--device\s+[^\s\"']+":"--device cuda",
    }
    counts=[]
    for p,r in replacements.items(): text,n=re.subn(p,r,text); counts.append(n)
    path.write_text(text); report.append(f"{path}: app_config replacements={counts}")
(job/"PATCH_REPORT.txt").write_text("\n".join(report)+"\n")
print("\n".join(report))
PY
}

extract_metrics() {
  local run_dir="$1" label="$2" seed="$3" clients="$4" cpr="$5" rounds="$6" epochs="$7" status="$8" exit_code="$9" duration="${10}"
  python - "$run_dir" "$SUMMARY_CSV" "$label" "$seed" "$clients" "$cpr" "$rounds" "$epochs" "$LR" "$BATCH_SIZE" "$MODE_NAME" "$status" "$exit_code" "$duration" <<'PY'
import csv, json, re, sys
from pathlib import Path
run_dir, summary, label, seed, clients, cpr, rounds, epochs, lr, bs, mode, status, exit_code, duration = sys.argv[1:]
run=Path(run_dir)
text=(run/"stdout.log").read_text(errors="ignore") if (run/"stdout.log").exists() else ""
central=(run/"central_eval.log").read_text(errors="ignore") if (run/"central_eval.log").exists() else ""
def vals(pattern, src=text): return [float(x) for x in re.findall(pattern, src, flags=re.I)]
# Preferimos evaluación centralizada final si existe.
accs=vals(r"final_accuracy=([0-9.eE+-]+)",central) or vals(r"\[NVFLARE METRICS\].*?accuracy=([0-9.eE+-]+)")
losses=vals(r"final_loss=([0-9.eE+-]+)",central) or vals(r"\[NVFLARE METRICS\].*?val_loss=([0-9.eE+-]+)")
train_losses=vals(r"\[NVFLARE METRICS\].*?train_loss=([0-9.eE+-]+)")
with (run/"metrics_extracted.csv").open("w",newline="") as f:
    w=csv.writer(f); w.writerow(["event_index","train_loss","val_loss","accuracy","num_examples","steps"])
    for i,m in enumerate(re.finditer(r"\[NVFLARE METRICS\]\s+train_loss=([0-9.eE+-]+)\s+val_loss=([0-9.eE+-]+)\s+accuracy=([0-9.eE+-]+)\s+num_examples=(\d+)\s+steps=(\d+)",text,re.I)):
        w.writerow([i,*m.groups()])
def num(s):
    m=re.search(r"[-+]?[0-9]*\.?[0-9]+",str(s)); return float(m.group()) if m else None
gpu=[]
try:
    with (run/"gpu.csv").open(errors="ignore") as f: gpu=list(csv.DictReader(f))
except Exception: pass
def cv(fragment):
    out=[]
    for row in gpu:
        for k,v in row.items():
            if fragment.lower() in k.lower():
                x=num(v)
                if x is not None: out.append(x)
                break
    return out
mem=cv("memory.used"); util=cv("utilization.gpu"); power=cv("power.draw"); temp=cv("temperature.gpu")
time_text=(run/"time.txt").read_text(errors="ignore") if (run/"time.txt").exists() else ""
m=re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)",time_text); max_rss=int(m.group(1)) if m else None
ci=int(clients); cp=int(cpr); ro=int(rounds); ep=int(epochs); dur=float(duration); updates=cp*ro; estimated=int(40000*(cp/ci)*ep*ro)
energy=(sum(power)/len(power))*dur/3600 if power and dur>0 else None
row={"framework":"nvflare","mode":mode,"experiment":label,"seed":int(seed),"clients":ci,"clients_per_round":cp,"rounds":ro,"local_epochs":ep,"batch_size":int(bs),"learning_rate":float(lr),"status":status,"exit_code":int(exit_code),"duration_seconds":dur,"client_updates":updates,"estimated_train_examples":estimated,"estimated_examples_per_second":estimated/dur if dur>0 else None,"final_accuracy":accs[-1] if accs else None,"final_loss":losses[-1] if losses else None,"final_train_accuracy":None,"final_train_loss":train_losses[-1] if train_losses else None,"gpu_mem_max_mib":max(mem) if mem else None,"gpu_mem_mean_mib":sum(mem)/len(mem) if mem else None,"gpu_util_max_pct":max(util) if util else None,"gpu_util_mean_pct":sum(util)/len(util) if util else None,"power_max_w":max(power) if power else None,"power_mean_w":sum(power)/len(power) if power else None,"estimated_energy_wh":energy,"temp_max_c":max(temp) if temp else None,"max_rss_kb":max_rss,"run_dir":str(run.resolve())}
(run/"summary.json").write_text(json.dumps(row,indent=2))
with open(summary,"a",newline="") as f: csv.DictWriter(f,fieldnames=list(row.keys())).writerow(row)
PY
}

if [[ "$MODE" == "1" ]]; then
  echo "Modo 1: -t 1 limita NVIDIA FLARE a una tarea de entrenamiento activa. Los sitios pueden permanecer creados y reutilizarse."
else
  echo "Modo 2: -t clientes_por_ronda intenta ejecutar tareas concurrentes. Puede producir OOM."
fi
echo "AVISO E3: min_clients=4 representa participación parcial, pero revisa en el log cuántos clientes respondieron realmente."
echo "Suite: $SUITE_DIR"

for spec in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r label clients cpr min_clients rounds epochs <<< "$spec"
  [[ "$label" == "e0_baseline" && "$INCLUDE_BASELINE" != "1" ]] && continue
  [[ -n "$EXPERIMENT_FILTER" && "$label" != *"$EXPERIMENT_FILTER"* ]] && continue
  for ((rep=0; rep<REPEATS; rep++)); do
    seed=$((42+rep))
    threads=$([[ "$MODE" == "1" ]] && echo 1 || echo "$cpr")
    run_name="nvflare_${label}_${TFG_FL_STRATEGY}_${MODE_NAME}_clients${clients}_cpr${cpr}_rounds${rounds}_ep${epochs}_lr002_bs32_seed${seed}"
    run_dir="$SUITE_DIR/$run_name"; job_dir="$run_dir/job"; workspace="$run_dir/workspace"
    mkdir -p "$run_dir"; snapshot_environment "$run_dir"
    cp -a "$BASE_JOB_DIR" "$job_dir"

    # IMPORTANTE:
    # El job copiado puede llevar una copia antigua de client.py/task.py.
    # Forzamos que el job use siempre el código actual de PROJECT_DIR/code.
    python - "$PROJECT_DIR" "$job_dir" <<'PYJOB'
import sys
from pathlib import Path

project = Path(sys.argv[1])
job = Path(sys.argv[2])

for name in ["client.py", "task.py", "eval_final.py"]:
    src = project / "code" / name
    if not src.exists():
        print(f"[COPY CODE] no existe {src}")
        continue

    matches = [p for p in job.rglob(name) if p.is_file()]

    if not matches:
        print(f"[COPY CODE] aviso: no encuentro {name} dentro del job")
        continue

    for dst in matches:
        dst.write_bytes(src.read_bytes())
        print(f"[COPY CODE] {src} -> {dst}")
PYJOB

    patch_job "$job_dir" "$clients" "$min_clients" "$rounds" "$epochs" "$seed" | tee "$run_dir/config/patch_job.log"

    # Parche FedOpt real en servidor para FedAdam/FedAdagrad.
    python - "$job_dir" "$TFG_FL_STRATEGY" <<'PYFEDOPT'
import sys
from pathlib import Path

job_dir = Path(sys.argv[1])
strategy = sys.argv[2].lower()

server_files = list(job_dir.rglob("config_fed_server.conf"))

if not server_files:
    raise SystemExit("No encuentro config_fed_server.conf dentro del job")

if strategy == "fedadam":
    optimizer = "torch.optim.Adam"
elif strategy == "fedadagrad":
    optimizer = "torch.optim.Adagrad"
elif strategy in ("fedavg", "fedprox"):
    print(f"[FEDOPT PATCH] strategy={strategy}; no cambio shareable_generator")
    raise SystemExit(0)
else:
    raise SystemExit(f"Estrategia no soportada: {strategy}")

old_block = "\n".join([
    "    {",
    '      id = "shareable_generator"',
    '      path = "nvflare.app_common.shareablegenerators.full_model_shareable_generator.FullModelShareableGenerator"',
    "      args {}",
    "    }",
])

new_block = "\n".join([
    "    {",
    '      id = "shareable_generator"',
    '      path = "nvflare.app_opt.pt.fedopt.PTFedOptModelShareableGenerator"',
    "      args {",
    '        device = "cpu"',
    '        source_model = "model"',
    "        optimizer_args {",
    f'          path = "{optimizer}"',
    "          args {",
    "            lr = 0.01",
    "            foreach = false",
    "            fused = false",
    "            capturable = false",
    "          }",
    '          config_type = "dict"',
    "        }",
    "      }",
    "    }",
])

for path in server_files:
    txt = path.read_text()

    if "PTFedOptModelShareableGenerator" in txt:
        print(f"[FEDOPT PATCH] ya estaba aplicado en {path}")
        continue

    if old_block not in txt:
        raise SystemExit(f"No encuentro el bloque shareable_generator esperado en {path}")

    txt = txt.replace(old_block, new_block, 1)
    path.write_text(txt)

    print(f"[FEDOPT PATCH] {path}: strategy={strategy}, optimizer={optimizer}")
PYFEDOPT

    # Parche componente model requerido por FedOpt.
    python - "$job_dir" "$TFG_FL_STRATEGY" <<'PYMODEL'
import sys
from pathlib import Path

job_dir = Path(sys.argv[1])
strategy = sys.argv[2].lower()

if strategy not in ("fedadam", "fedadagrad"):
    raise SystemExit(0)

server_files = list(job_dir.rglob("config_fed_server.conf"))

if not server_files:
    raise SystemExit("No encuentro config_fed_server.conf dentro del job")

model_block = "\n".join([
    "    {",
    '      id = "model"',
    '      path = "{model_class_path}"',
    "      args {}",
    "    }",
])

for path in server_files:
    txt = path.read_text()

    if 'id = "model"' in txt:
        print(f"[MODEL COMPONENT PATCH] ya existe id=model en {path}")
        continue

    target = "  components = [\n"
    if target not in txt:
        raise SystemExit(f"No encuentro components = [ en {path}")

    txt = txt.replace(target, target + model_block + "\n", 1)
    path.write_text(txt)

    print(f"[MODEL COMPONENT PATCH] añadido id=model en {path}")
PYMODEL
    cp -a "$job_dir"/app/config "$run_dir/config/job_config_snapshot" 2>/dev/null || true
    cat > "$run_dir/config/experiment.txt" <<CFG
clients=$clients
clients_per_round_requested=$cpr
min_clients=$min_clients
rounds=$rounds
local_epochs=$epochs
batch_size=$BATCH_SIZE
learning_rate=$LR
seed=$seed
strategy=$TFG_FL_STRATEGY
fedprox_mu=$TFG_FEDPROX_MU
simulator_threads=$threads
CFG
    echo "===== $run_name ====="
    start_monitors "$run_dir"; start_epoch=$(date +%s)
    set +e
    PYTHONPATH="$SUITE_DIR:${PYTHONPATH:-}" TFG_SEED="$seed" PYTHONHASHSEED="$seed" TFG_FL_STRATEGY="$TFG_FL_STRATEGY" TFG_FEDPROX_MU="$TFG_FEDPROX_MU" \
      /usr/bin/time -v -o "$run_dir/time.txt" \
      nvflare simulator -w "$workspace" -n "$clients" -t "$threads" ${TFG_SIM_GPU---gpu 0} "$job_dir" \
      2>&1 | tee "$run_dir/stdout.log"
    exit_code=${PIPESTATUS[0]}
    end_epoch=$(date +%s); duration=$((end_epoch-start_epoch)); stop_monitors

    model_file=""
    while IFS= read -r candidate; do model_file="$candidate"; break; done < <(find "$workspace" -type f -name 'FL_global_model.pt' 2>/dev/null)
    if [[ -z "$model_file" ]]; then while IFS= read -r candidate; do model_file="$candidate"; break; done < <(find "$workspace" -type f -name 'best_FL_global_model.pt' 2>/dev/null); fi
    if [[ -n "$model_file" ]]; then
      cp "$model_file" "$run_dir/final_model.pt"
      if [[ -f "$PROJECT_DIR/code/eval_final.py" ]]; then
        PYTHONPATH="$SUITE_DIR:${PYTHONPATH:-}" TFG_SEED="$seed" python "$PROJECT_DIR/code/eval_final.py" --model_path "$run_dir/final_model.pt" --device cuda > "$run_dir/central_eval.log" 2>&1 || true
      fi
    fi
    bad_patterns="FATAL_SYSTEM_ERROR|TASK_ABORTED|Cannot reach min_responses|Traceback|failed to send|failed to download object|External process has not called flare.init|Launcher already exited|TimeoutError|timed out"

    last_finished_round=$(
      grep -E "Round [0-9]+ finished" "$run_dir/stdout.log" 2>/dev/null \
      | sed -E 's/.*Round ([0-9]+) finished.*/\1/' \
      | sort -n \
      | tail -1
    )

    expected_last_round=$((rounds - 1))

    if grep -Eiq "$bad_patterns" "$run_dir/stdout.log"; then
      status="failed"
    elif [[ -z "$last_finished_round" || "$last_finished_round" -lt "$expected_last_round" ]]; then
      status="incomplete"
    elif [[ "$exit_code" == "0" ]]; then
      status="success"
    else
      status="failed"
    fi

    extract_metrics "$run_dir" "$label" "$seed" "$clients" "$cpr" "$rounds" "$epochs" "$status" "$exit_code" "$duration"
    date --iso-8601=seconds > "$run_dir/environment/end_time.txt"
    echo "$run_name -> $status (exit=$exit_code, ${duration}s)"
    sleep "$COOLDOWN_SECONDS"
  done
done

echo "Finalizado. Resumen: $SUMMARY_CSV"
