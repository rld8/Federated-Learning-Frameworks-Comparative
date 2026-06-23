#!/usr/bin/env bash
# Ejecuta la matriz completa de experimentos FedML del TFG usando ~/Escritorio/tfg/FedML/code.
# Uso: ./run_fedml_experiments.sh 1   # backend sp, una tarea de cliente activa cada vez
#      ./run_fedml_experiments.sh 2   # backend MPI, procesos de cliente concurrentes
# Variables opcionales: REPEATS=3 INCLUDE_BASELINE=1 EXPERIMENT_FILTER=e1 COOLDOWN_SECONDS=5

set -uo pipefail

MODE="${1:-}"
if [[ "$MODE" != "1" && "$MODE" != "2" ]]; then
  echo "Uso: $0 <modo>"
  echo "  modo 1: backend sp, entrenamiento secuencial"
  echo "  modo 2: backend MPI, clientes concurrentes"
  exit 1
fi

PROJECT_DIR="${FEDML_PROJECT_DIR:-$HOME/Escritorio/tfg/FedML}"
CODE_DIR="${FEDML_CODE_DIR:-$PROJECT_DIR/code}"
REPEATS="${REPEATS:-3}"
INCLUDE_BASELINE="${INCLUDE_BASELINE:-1}"
EXPERIMENT_FILTER="${EXPERIMENT_FILTER:-}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-5}"
LR="${LR:-0.02}"
BATCH_SIZE="${BATCH_SIZE:-32}"
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
)

mkdir -p "$SUITE_DIR"
echo 'framework,mode,experiment,seed,clients,clients_per_round,rounds,local_epochs,batch_size,learning_rate,status,exit_code,duration_seconds,client_updates,estimated_train_examples,estimated_examples_per_second,final_accuracy,final_loss,final_train_accuracy,final_train_loss,gpu_mem_max_mib,gpu_mem_mean_mib,gpu_util_max_pct,gpu_util_mean_pct,power_max_w,power_mean_w,estimated_energy_wh,temp_max_c,max_rss_kb,run_dir' > "$SUMMARY_CSV"

if [[ ! -d "$CODE_DIR" ]]; then echo "ERROR: no existe CODE_DIR=$CODE_DIR"; exit 1; fi
cd "$CODE_DIR"
for cmd in python nvidia-smi /usr/bin/time; do command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: falta $cmd"; exit 1; }; done
if [[ "$MODE" == "2" ]]; then command -v mpirun >/dev/null 2>&1 || { echo "ERROR: falta mpirun"; exit 1; }; fi

if ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi no funciona. No ejecutes experimentos porque las métricas GPU no serán válidas."
  nvidia-smi || true
  exit 1
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
  cp -a "$CODE_DIR"/*.py "$dir/code/" 2>/dev/null || true
}

MONITOR_PIDS=()
start_monitors() {
  local dir="$1"
  nvidia-smi --query-gpu=timestamp,index,name,pstate,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem --format=csv,nounits -l 1 > "$dir/gpu.csv" 2>&1 & MONITOR_PIDS+=("$!")
  nvidia-smi dmon -s pucvmet -d 1 > "$dir/gpu_dmon.txt" 2>&1 & MONITOR_PIDS+=("$!")
  nvidia-smi pmon -s um -d 1 > "$dir/gpu_pmon.txt" 2>&1 & MONITOR_PIDS+=("$!")
  vmstat -t 1 > "$dir/vmstat.txt" 2>&1 & MONITOR_PIDS+=("$!")
  (while true; do echo "===== $(date --iso-8601=seconds) ====="; ps -eo pid,ppid,%cpu,%mem,rss,vsz,etimes,cmd --sort=-rss | head -100; sleep 1; done) > "$dir/process_snapshots.txt" 2>&1 & MONITOR_PIDS+=("$!")
}
stop_monitors() { local pid; for pid in "${MONITOR_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done; wait "${MONITOR_PIDS[@]:-}" 2>/dev/null || true; MONITOR_PIDS=(); }

generate_config() {
  local config="$1" mapping="$2" clients="$3" cpr="$4" rounds="$5" epochs="$6" seed="$7" backend="$8"
  local mapping_lines=""
  if [[ "$backend" == "MPI" ]]; then
    mapping_lines="  gpu_mapping_file: \"$mapping\"\n  gpu_mapping_key: \"mapping_run\""
  fi
  printf '%s\n' \
'common_args:' \
'  training_type: "simulation"' \
"  random_seed: $seed" \
'  using_mlops: false' \
'' \
'data_args:' \
'  dataset: "cifar10"' \
'  data_cache_dir: "~/Escritorio/tfg/NFLARE/downloads"' \
'  partition_method: "iid"' \
'  partition_alpha: 0.5' \
'' \
'model_args:' \
'  model: "resnet18_cifar10"' \
'' \
'train_args:' \
'  federated_optimizer: "FedAvg"' \
'  client_id_list: "[]"' \
"  client_num_in_total: $clients" \
"  client_num_per_round: $cpr" \
"  comm_round: $rounds" \
"  epochs: $epochs" \
"  batch_size: $BATCH_SIZE" \
'  client_optimizer: "sgd"' \
"  learning_rate: $LR" \
"  lr: $LR" \
'  weight_decay: 0.0005' \
'' \
'validation_args:' \
'  frequency_of_the_test: 1' \
'' \
'device_args:' \
'  using_gpu: true' \
'  gpu_id: 0' > "$config"
  if [[ -n "$mapping_lines" ]]; then printf '%b\n' "$mapping_lines" >> "$config"; fi
  printf '%s\n' '' 'comm_args:' "  backend: \"$backend\"" '' 'tracking_args:' '  log_file_dir: "./log"' '  enable_wandb: false' >> "$config"
}

extract_metrics() {
  local run_dir="$1" label="$2" seed="$3" clients="$4" cpr="$5" rounds="$6" epochs="$7" status="$8" exit_code="$9" duration="${10}"
  python - "$run_dir" "$SUMMARY_CSV" "$label" "$seed" "$clients" "$cpr" "$rounds" "$epochs" "$LR" "$BATCH_SIZE" "$MODE_NAME" "$status" "$exit_code" "$duration" <<'PY'
import csv, json, re, sys
from pathlib import Path
run_dir, summary, label, seed, clients, cpr, rounds, epochs, lr, bs, mode, status, exit_code, duration = sys.argv[1:]
run=Path(run_dir); text=(run/"stdout.log").read_text(errors="ignore") if (run/"stdout.log").exists() else ""
def vals(p): return [float(x) for x in re.findall(p,text,flags=re.I)]
accs=vals(r"[\'\"]test_acc[\'\"]\s*:\s*([0-9.eE+-]+)") or vals(r"accuracy=([0-9.eE+-]+)")
losses=vals(r"[\'\"]test_loss[\'\"]\s*:\s*([0-9.eE+-]+)") or vals(r"loss=([0-9.eE+-]+)")
train_accs=vals(r"['\"]training_acc['\"]\s*:\s*([0-9.eE+-]+)")
train_losses=vals(r"[\'\"]training_loss[\'\"]\s*:\s*([0-9.eE+-]+)") or vals(r"train_loss=([0-9.eE+-]+)") or vals(r"train_loss=([0-9.eE+-]+)")
with (run/"metrics_extracted.csv").open("w",newline="") as f:
    w=csv.writer(f); w.writerow(["event_index","training_acc","training_loss","test_acc","test_loss"])
    train=list(re.finditer(r"\{['\"]training_acc['\"]:\s*([0-9.eE+-]+),\s*['\"]training_loss['\"]:\s*([0-9.eE+-]+)\}",text,re.I))
    test=list(re.finditer(r"\{['\"]test_acc['\"]:\s*([0-9.eE+-]+),\s*['\"]test_loss['\"]:\s*([0-9.eE+-]+)\}",text,re.I))
    for i in range(max(len(train),len(test))):
        ta,tl=(train[i].groups() if i<len(train) else (None,None)); va,vl=(test[i].groups() if i<len(test) else (None,None)); w.writerow([i,ta,tl,va,vl])
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
time_text=(run/"time.txt").read_text(errors="ignore") if (run/"time.txt").exists() else ""; m=re.search(r"MAX_RSS_KB=(\d+)",time_text) or re.search(r"Maximum resident set size \\(kbytes\\):\\s*(\\d+)",time_text); max_rss=int(m.group(1)) if m else None
ci=int(clients); cp=int(cpr); ro=int(rounds); ep=int(epochs); dur=float(duration); updates=cp*ro; estimated=int(40000*(cp/ci)*ep*ro); energy=(sum(power)/len(power))*dur/3600 if power and dur>0 else None
row={"framework":"fedml","mode":mode,"experiment":label,"seed":int(seed),"clients":ci,"clients_per_round":cp,"rounds":ro,"local_epochs":ep,"batch_size":int(bs),"learning_rate":float(lr),"status":status,"exit_code":int(exit_code),"duration_seconds":dur,"client_updates":updates,"estimated_train_examples":estimated,"estimated_examples_per_second":estimated/dur if dur>0 else None,"final_accuracy":accs[-1] if accs else None,"final_loss":losses[-1] if losses else None,"final_train_accuracy":train_accs[-1] if train_accs else None,"final_train_loss":train_losses[-1] if train_losses else None,"gpu_mem_max_mib":max(mem) if mem else None,"gpu_mem_mean_mib":sum(mem)/len(mem) if mem else None,"gpu_util_max_pct":max(util) if util else None,"gpu_util_mean_pct":sum(util)/len(util) if util else None,"power_max_w":max(power) if power else None,"power_mean_w":sum(power)/len(power) if power else None,"estimated_energy_wh":energy,"temp_max_c":max(temp) if temp else None,"max_rss_kb":max_rss,"run_dir":str(run.resolve())}
(run/"summary.json").write_text(json.dumps(row,indent=2));
with open(summary,"a",newline="") as f: csv.DictWriter(f,fieldnames=list(row.keys())).writerow(row)
PY
}

if [[ "$MODE" == "1" ]]; then
  echo "Modo 1: FedML backend sp usa un único proceso y entrena clientes secuencialmente. FedML puede reutilizar objetos cliente; no garantiza destruirlos físicamente tras cada turno."
else
  echo "Modo 2: FedML backend MPI mantiene procesos de cliente concurrentes. En una GPU de 6 GB puede producir OOM; el fallo se registrará y la suite continuará."
fi
echo "Suite: $SUITE_DIR"

for spec in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r label clients cpr rounds epochs <<< "$spec"
  [[ "$label" == "e0_baseline" && "$INCLUDE_BASELINE" != "1" ]] && continue
  [[ -n "$EXPERIMENT_FILTER" && "$label" != *"$EXPERIMENT_FILTER"* ]] && continue
  for ((rep=0; rep<REPEATS; rep++)); do
    seed=$((42+rep))
    backend=$([[ "$MODE" == "1" ]] && echo "sp" || echo "MPI")
    run_name="fedml_${label}_${MODE_NAME}_clients${clients}_cpr${cpr}_rounds${rounds}_ep${epochs}_lr002_bs32_seed${seed}"
    run_dir="$SUITE_DIR/$run_name"; config="$run_dir/config/fedml_config.yaml"; mapping="$run_dir/config/gpu_mapping.yaml"
    mkdir -p "$run_dir"; snapshot_environment "$run_dir"
    if [[ "$backend" == "MPI" ]]; then
      cat > "$mapping" <<MAP
mapping_run:
  "$(hostname)": [$((cpr+1))]
MAP
    fi
    generate_config "$config" "$mapping" "$clients" "$cpr" "$rounds" "$epochs" "$seed" "$backend"
    echo "===== $run_name ====="
    rm -f final_model_fedml_common.pt
    start_monitors "$run_dir"; start_epoch=$(date +%s)
    set +e
    if [[ "$backend" == "sp" ]]; then
      PYTHONPATH="$SUITE_DIR:${PYTHONPATH:-}" TFG_SEED="$seed" PYTHONHASHSEED="$seed" \
        /usr/bin/time -f "ELAPSED_SECONDS=%e\nMAX_RSS_KB=%M\nUSER_SECONDS=%U\nSYSTEM_SECONDS=%S\nCPU_PERCENT=%P" -o "$run_dir/time.txt" python main.py --cf "$config" 2>&1 | tee "$run_dir/stdout.log"
      exit_code=${PIPESTATUS[0]}
    else
      PYTHONPATH="$SUITE_DIR:${PYTHONPATH:-}" TFG_SEED="$seed" PYTHONHASHSEED="$seed" \
        /usr/bin/time -f "ELAPSED_SECONDS=%e\nMAX_RSS_KB=%M\nUSER_SECONDS=%U\nSYSTEM_SECONDS=%S\nCPU_PERCENT=%P" -o "$run_dir/time.txt" mpirun --oversubscribe -np "$((cpr+1))" python main.py --cf "$config" 2>&1 | tee "$run_dir/stdout.log"
      exit_code=${PIPESTATUS[0]}
    fi
    end_epoch=$(date +%s); duration=$((end_epoch-start_epoch)); stop_monitors
    [[ -f final_model_fedml_common.pt ]] && cp final_model_fedml_common.pt "$run_dir/final_model.pt"
    status=$([[ "$exit_code" == "0" ]] && echo success || echo failed)
    extract_metrics "$run_dir" "$label" "$seed" "$clients" "$cpr" "$rounds" "$epochs" "$status" "$exit_code" "$duration"
    date --iso-8601=seconds > "$run_dir/environment/end_time.txt"
    echo "$run_name -> $status (exit=$exit_code, ${duration}s)"
    sleep "$COOLDOWN_SECONDS"
  done
done

echo "Finalizado. Resumen: $SUMMARY_CSV"
