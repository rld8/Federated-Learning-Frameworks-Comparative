import torch

from fedml.core import ClientTrainer
from task import train


class FedMLCifar10Trainer(ClientTrainer):
    def __init__(self, model, args=None):
        super().__init__(model, args)
        self.model = model
        self.args = args

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        self.model.to(device)

        federated_optimizer = str(getattr(args, "federated_optimizer", "FedAvg"))
        strategy_name = str(getattr(args, "strategy_name_for_tfg", federated_optimizer))

        fedprox_mu = 0.0
        global_params = None

        if federated_optimizer.lower() == "fedprox" or strategy_name.lower() == "fedprox":
            fedprox_mu = float(getattr(args, "fedprox_mu", 0.001))
            global_params = {
                name: param.detach().clone().cpu()
                for name, param in self.model.named_parameters()
            }

            print(f"[FEDML TRAIN] FedProx activo con mu={fedprox_mu}")
        else:
            print(f"[FEDML TRAIN] estrategia local={strategy_name}")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            print(f"[FEDML TRAIN] GPU={torch.cuda.get_device_name(0)}")
            print(
                "[FEDML TRAIN] allocated_mb_before="
                f"{torch.cuda.memory_allocated(0) / 1024 / 1024:.2f}"
            )

        avg_loss = train(
            model=self.model,
            trainloader=train_data,
            epochs=int(args.epochs),
            lr=float(args.learning_rate),
            device=device,
            weight_decay=float(args.weight_decay),
            fedprox_mu=fedprox_mu,
            global_params=global_params,
        )

        if torch.cuda.is_available():
            print(
                "[FEDML TRAIN] allocated_mb_after="
                f"{torch.cuda.memory_allocated(0) / 1024 / 1024:.2f}"
            )
            print(
                "[FEDML TRAIN] peak_allocated_mb="
                f"{torch.cuda.max_memory_allocated(0) / 1024 / 1024:.2f}"
            )

        print(f"[FEDML METRICS] train_loss={avg_loss:.4f}")
