from pathlib import Path
import re

server_app = Path(__import__("os").environ["SERVER_APP"])
s = server_app.read_text()

if "FedAdagrad" not in s:
    s = s.replace(
        "from flwr.serverapp.strategy import FedAvg",
        "from flwr.serverapp.strategy import FedAvg, FedAdagrad, FedAdam"
    )

if "strategy_name" in s and "fedadagrad" in s and "fedadam" in s:
    print("El soporte de estrategias ya parece estar añadido.")
    server_app.write_text(s)
    raise SystemExit(0)

pattern = re.compile(r"(\s*)strategy\s*=\s*FedAvg\s*\(", re.MULTILINE)
m = pattern.search(s)

if not m:
    raise SystemExit(
        "No he encontrado una línea del tipo 'strategy = FedAvg('. "
        "No modifico nada para evitar romper el archivo."
    )

indent = m.group(1)

replacement = (
    f'{indent}strategy_name = str(context.run_config.get("strategy", "fedavg")).lower()\\n'
    f'{indent}print(f"[FLOWER STRATEGY] strategy={{strategy_name}}")\\n'
    f'{indent}\\n'
    f'{indent}if strategy_name == "fedadagrad":\\n'
    f'{indent}    strategy_class = FedAdagrad\\n'
    f'{indent}elif strategy_name == "fedadam":\\n'
    f'{indent}    strategy_class = FedAdam\\n'
    f'{indent}else:\\n'
    f'{indent}    strategy_class = FedAvg\\n'
    f'{indent}\\n'
    f'{indent}strategy = strategy_class('
)

s = pattern.sub(replacement, s, count=1)
server_app.write_text(s)

print(f"Parche aplicado en {server_app}")
