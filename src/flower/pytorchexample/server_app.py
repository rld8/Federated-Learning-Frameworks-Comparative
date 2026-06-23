"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedAdagrad, FedAdam

from pytorchexample.task import Net, load_centralized_dataset, test


# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    fraction_train: float = float(context.run_config.get("fraction-train", 0.5))
    fraction_evaluate: float = float(context.run_config.get("fraction-evaluate", 0.0))

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Select strategy
    strategy_name = str(context.run_config.get("strategy", "fedavg")).lower()
    print(f"[FLOWER STRATEGY] strategy={strategy_name}")

    common_kwargs = dict(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=1,
        min_evaluate_nodes=0,
        min_available_nodes=1,
    )

    if strategy_name == "fedadagrad":
        strategy_class = FedAdagrad
        adaptive_kwargs = dict(
            eta=float(context.run_config.get("server-lr", 0.01)),
            tau=float(context.run_config.get("tau", 0.01)),
        )
    elif strategy_name == "fedadam":
        strategy_class = FedAdam
        adaptive_kwargs = dict(
            eta=float(context.run_config.get("server-lr", 0.01)),
            tau=float(context.run_config.get("tau", 0.01)),
        )
    else:
        strategy_class = FedAvg
        adaptive_kwargs = {}

    if adaptive_kwargs:
        print(f"[FLOWER STRATEGY] adaptive_kwargs={adaptive_kwargs}")

    strategy = strategy_class(**common_kwargs, **adaptive_kwargs)

    # Start strategy, run for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate global model on the centralized CIFAR-10 test set."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the full CIFAR-10 test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model
    test_loss, test_acc = test(model, test_dataloader, device)

    print(
        f"Server-side evaluation round {server_round}: "
        f"loss={test_loss:.4f}, accuracy={test_acc:.4f}"
    )

    # Return the evaluation metrics
    return MetricRecord(
        {
            "accuracy": test_acc,
            "loss": test_loss,
        }
    )
