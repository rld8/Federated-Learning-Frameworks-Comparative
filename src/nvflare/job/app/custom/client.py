import argparse
import os

import torch
import nvflare.client as flare



try:
    FLModel = flare.FLModel
except AttributeError:
    from nvflare.app_common.abstract.fl_model import FLModel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def get_site_name():
    for name in ["NVFLARE_SITE_NAME", "SITE_NAME"]:
        value = os.environ.get(name)
        if value:
            return value

    try:
        return flare.get_site_name()
    except Exception:
        return "site-1"


def get_partition_id(site_name):
    try:
        return int(site_name.split("-")[-1]) - 1
    except Exception:
        return 0


def load_global_model(model, input_model):
    if input_model is None:
        return

    if input_model.params is None:
        return

    current_state = model.state_dict()
    new_state = {}

    for name, value in input_model.params.items():
        if name not in current_state:
            continue

        if torch.is_tensor(value):
            tensor = value
        else:
            tensor = torch.tensor(value)

        tensor = tensor.to(dtype=current_state[name].dtype)
        new_state[name] = tensor

    current_state.update(new_state)
    model.load_state_dict(current_state, strict=False)


def get_model_params(model):
    params = {}

    for name, value in model.state_dict().items():
        params[name] = value.detach().cpu()

    return params


def main():
    args = parse_args()

    torch.set_num_threads(2)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    flare.init()

    from task import Net, load_data, train, test

    site_name = get_site_name()
    partition_id = get_partition_id(site_name)

    print(f"[NVFLARE CLIENT] site_name={site_name}")
    print(f"[NVFLARE CLIENT] partition_id={partition_id}")
    print(f"[NVFLARE CLIENT] device={device}")

    trainloader, valloader = load_data(
        partition_id=partition_id,
        num_partitions=args.num_clients,
        batch_size=args.batch_size,
    )

    model = Net().to(device)

    input_model = flare.receive()

    while input_model is not None:
        load_global_model(model, input_model)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(device)
            allocated_before = torch.cuda.memory_allocated(device) / 1024 / 1024
            print(f"[NVFLARE TRAIN] GPU={gpu_name}")
            print(f"[NVFLARE TRAIN] allocated_mb_before={allocated_before:.2f}")

        train_loss = train(
            model=model,
            trainloader=trainloader,
            epochs=args.local_epochs,
            lr=args.lr,
            device=device,
        )

        val_loss, accuracy = test(
            model=model,
            testloader=valloader,
            device=device,
        )

        if torch.cuda.is_available():
            allocated_after = torch.cuda.memory_allocated(device) / 1024 / 1024
            peak_allocated = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            print(f"[NVFLARE TRAIN] allocated_mb_after={allocated_after:.2f}")
            print(f"[NVFLARE TRAIN] peak_allocated_mb={peak_allocated:.2f}")

        num_examples = len(trainloader.dataset)
        steps = len(trainloader) * args.local_epochs

        metrics = {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "accuracy": float(accuracy),
            "num-examples": int(num_examples),
        }

        print(
            "[NVFLARE METRICS] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"accuracy={accuracy:.4f} "
            f"num_examples={num_examples} "
            f"steps={steps}"
        )

        output_model = FLModel(
            params=get_model_params(model),
            metrics=metrics,
            meta={
                "NUM_STEPS_CURRENT_ROUND": steps,
            },
        )

        flare.send(output_model)

        input_model = flare.receive()

    flare.shutdown()


if __name__ == "__main__":
    main()
