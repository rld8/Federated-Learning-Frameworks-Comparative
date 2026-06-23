import torch

from fedml.core import ServerAggregator
from task import test


class FedMLCifar10Aggregator(ServerAggregator):
    def __init__(self, model, args=None):
        super().__init__(model, args)
        self.model = model
        self.args = args

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def test(self, test_data, device, args):
        self.model.to(device)

        loss, accuracy = test(
            model=self.model,
            testloader=test_data,
            device=device,
        )

        total = len(test_data.dataset)
        correct = int(accuracy * total)

        metrics = {
            "test_correct": correct,
            "test_loss": loss * total,
            "test_total": total,
            "accuracy": accuracy,
            "loss": loss,
        }

        print(
            "[FEDML GLOBAL TEST] "
            f"loss={loss:.4f} "
            f"accuracy={accuracy:.4f} "
            f"correct={correct} "
            f"total={total}"
        )

        return metrics

    def test_all(
        self,
        train_data_local_dict,
        test_data_local_dict,
        device,
        args=None,
    ):
        return True
