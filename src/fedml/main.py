import random
import numpy as np
import torch
import fedml

from fedml import FedMLRunner

from task import Net, load_partition_data_cifar10
from fedml_trainer import FedMLCifar10Trainer
from fedml_aggregator import FedMLCifar10Aggregator


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = fedml.init()

    seed = int(getattr(args, "random_seed", 42))
    set_seed(seed)

    torch.set_num_threads(2)

    device = fedml.device.get_device(args)

    process_id = int(getattr(args, "process_id", 0))

    print("[FEDML COMMON MAIN] inicializado")
    print(f"[FEDML COMMON MAIN] process_id={process_id}")
    print(f"[FEDML COMMON MAIN] seed={seed}")
    print(f"[FEDML COMMON MAIN] device={device}")
    print(f"[FEDML COMMON MAIN] cuda_available={torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"[FEDML COMMON MAIN] GPU={torch.cuda.get_device_name(0)}")

    dataset, output_dim = load_partition_data_cifar10(args)

    model = Net()

    trainer = FedMLCifar10Trainer(model, args)
    aggregator = FedMLCifar10Aggregator(model, args)

    runner = FedMLRunner(
        args=args,
        device=device,
        dataset=dataset,
        model=model,
        client_trainer=trainer,
        server_aggregator=aggregator,
    )

    runner.run()

    if process_id == 0:
        torch.save(model.state_dict(), "final_model_fedml_common.pt")
        print("[FEDML COMMON MAIN] modelo guardado en final_model_fedml_common.pt")
