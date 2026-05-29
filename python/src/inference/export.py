"""Export trained PyTorch model to ONNX for C++ inference."""

import os
import torch

from trainer.config import config
from trainer.model import SholoGutiNet


def export_onnx(model_path: str = "checkpoints/best_model.pt", output_path: str = "checkpoints/model.onnx"):
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return

    device = torch.device("cpu")  # Export from CPU

    # Initialize model
    model = SholoGutiNet(
        input_features=config.input_features,
        hidden_dim=config.hidden_dim,
        num_residual_blocks=config.num_residual_blocks,
        action_space=config.action_space,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Create dummy input: batch size 1, 80 features
    dummy_input = torch.zeros(1, config.input_features).to(device)

    # Export to ONNX
    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        export_params=True,
        opset_version=14,  # standard compatible opset
        do_constant_folding=True,
        input_names=["input"],  # input name
        output_names=["policy", "value"],  # output names
        dynamic_axes={"input": {0: "batch_size"}, "policy": {0: "batch_size"}, "value": {0: "batch_size"}},
    )
    print("Export complete!")


if __name__ == "__main__":
    export_onnx()
