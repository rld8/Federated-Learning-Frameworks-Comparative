# code/eval_final.py

import argparse
import torch

from task import Net, load_centralized_dataset, test


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def extract_state_dict(checkpoint):
    """
    NVFLARE puede guardar el modelo de varias formas.
    A veces guarda directamente el state_dict.
    Otras veces guarda un diccionario con claves como:
      - model
      - train_conf

    Esta función extrae solo los pesos reales del modelo.
    """

    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"]

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

        if "params" in checkpoint:
            return checkpoint["params"]

    return checkpoint


def clean_state_dict(state_dict):
    """
    Limpia posibles formatos raros:
    - convierte valores a tensor si hiciera falta
    - elimina prefijo 'module.' si aparece
    """

    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        if not torch.is_tensor(value):
            value = torch.tensor(value)

        cleaned[new_key] = value

    return cleaned


def main():
    args = parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = Net().to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    state_dict = clean_state_dict(state_dict)

    model.load_state_dict(state_dict, strict=True)

    testloader = load_centralized_dataset(batch_size=64)

    loss, acc = test(model, testloader, device)

    print(f"final_loss={loss:.4f}")
    print(f"final_accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
