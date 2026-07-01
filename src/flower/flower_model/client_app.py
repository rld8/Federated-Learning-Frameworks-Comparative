"""pytorchexample: A Flower / PyTorch app."""

import os
import time

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

app = ClientApp()


def _verbose() -> bool:
    return os.environ.get("TFG_VERBOSE_CLIENT", "1") != "0"


def _log(text: str) -> None:
    if _verbose():
        print(text, flush=True)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _cuda_memory(prefix: str) -> None:
    """Print CUDA memory info without ever crashing the training."""
    if not _verbose():
        return

    if not torch.cuda.is_available():
        _log(f"{prefix} cuda_available=False")
        return

    try:
        device_index = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device_index) / 1024**2
        reserved = torch.cuda.memory_reserved(device_index) / 1024**2
        max_allocated = torch.cuda.max_memory_allocated(device_index) / 1024**2
        gpu_name = torch.cuda.get_device_name(device_index)
        _log(
            f"{prefix} gpu_index={device_index} gpu={gpu_name} "
            f"allocated_mb={allocated:.2f} "
            f"reserved_mb={reserved:.2f} "
            f"peak_allocated_mb={max_allocated:.2f}"
        )
    except Exception as exc:
        # Importantísimo: un log de GPU no puede romper el experimento.
        _log(f"{prefix} gpu_memory_log_failed={type(exc).__name__}: {exc}")


def _cuda_sync() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception as exc:
        _log(f"[CUDA SYNC WARNING] {type(exc).__name__}: {exc}")


def _get_partition_info(context: Context) -> tuple[int, int]:
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    return partition_id, num_partitions


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    start_time = time.time()

    partition_id, num_partitions = _get_partition_info(context)
    batch_size = int(context.run_config["batch-size"])
    local_epochs = int(context.run_config["local-epochs"])
    lr = float(msg.content["config"]["lr"])
    device = _get_device()

    _log(
        f"[CLIENT TRAIN START] "
        f"partition={partition_id}/{num_partitions} "
        f"device={device} "
        f"batch_size={batch_size} "
        f"local_epochs={local_epochs} "
        f"lr={lr}"
    )

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.to(device)

    _log(f"[CLIENT TRAIN DEVICE] torch_cuda_available={torch.cuda.is_available()}")
    _log(f"[CLIENT TRAIN DEVICE] model_device={next(model.parameters()).device}")
    _cuda_memory("[CLIENT TRAIN GPU BEFORE DATA]")

    trainloader, _ = load_data(partition_id, num_partitions, batch_size)
    num_examples = len(trainloader.dataset)

    _log(
        f"[CLIENT TRAIN DATA] "
        f"partition={partition_id} "
        f"num_examples={num_examples} "
        f"num_batches={len(trainloader)}"
    )

    _cuda_memory("[CLIENT TRAIN GPU BEFORE TRAIN]")

    train_loss = train_fn(
        model,
        trainloader,
        local_epochs,
        lr,
        device,
    )

    _cuda_sync()

    duration = time.time() - start_time
    _cuda_memory("[CLIENT TRAIN GPU AFTER TRAIN]")

    _log(
        f"[CLIENT TRAIN END] "
        f"partition={partition_id} "
        f"train_loss={float(train_loss):.6f} "
        f"duration_seconds={duration:.2f}"
    )

    # Mandamos el modelo de vuelta en CPU para reducir presión de memoria GPU.
    cpu_state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    model_record = ArrayRecord(cpu_state_dict)

    metrics = {
        "train_loss": float(train_loss),
        "num-examples": num_examples,
        "train_duration_seconds": float(duration),
    }

    if torch.cuda.is_available():
        try:
            metrics["gpu_peak_allocated_mb"] = float(
                torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024**2
            )
        except Exception:
            pass

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})

    del model
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        _cuda_memory("[CLIENT TRAIN GPU AFTER CLEANUP]")

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    start_time = time.time()

    partition_id, num_partitions = _get_partition_info(context)
    batch_size = int(context.run_config["batch-size"])
    device = _get_device()

    _log(
        f"[CLIENT EVAL START] "
        f"partition={partition_id}/{num_partitions} "
        f"device={device} "
        f"batch_size={batch_size}"
    )

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.to(device)

    _log(f"[CLIENT EVAL DEVICE] model_device={next(model.parameters()).device}")
    _cuda_memory("[CLIENT EVAL GPU BEFORE DATA]")

    _, valloader = load_data(partition_id, num_partitions, batch_size)
    num_examples = len(valloader.dataset)

    _log(
        f"[CLIENT EVAL DATA] "
        f"partition={partition_id} "
        f"num_examples={num_examples} "
        f"num_batches={len(valloader)}"
    )

    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    _cuda_sync()

    duration = time.time() - start_time
    _cuda_memory("[CLIENT EVAL GPU AFTER EVAL]")

    _log(
        f"[CLIENT EVAL END] "
        f"partition={partition_id} "
        f"eval_loss={float(eval_loss):.6f} "
        f"eval_acc={float(eval_acc):.6f} "
        f"duration_seconds={duration:.2f}"
    )

    metrics = {
        "eval_loss": float(eval_loss),
        "eval_acc": float(eval_acc),
        "num-examples": num_examples,
        "eval_duration_seconds": float(duration),
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    del model
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        _cuda_memory("[CLIENT EVAL GPU AFTER CLEANUP]")

    return Message(content=content, reply_to=msg)
