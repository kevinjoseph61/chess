"""
Export trained PyTorch model to ONNX format for browser inference
using ONNX Runtime Web (WebAssembly).
"""

import argparse
import torch
from pathlib import Path

from .model import AlphaZeroNet, NUM_INPUT_PLANES


def export_to_onnx(checkpoint_path: str, output_path: str, opset_version: int = 17):
    """
    Export a trained AlphaZeroNet checkpoint to ONNX format.

    Args:
        checkpoint_path: path to .pt checkpoint file
        output_path: path for the output .onnx file
        opset_version: ONNX opset version (17 recommended for ONNX Runtime Web)
    """
    # Load model
    model = AlphaZeroNet()
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    # Support both checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Dummy input: batch=1, 18 planes, 8x8
    dummy_input = torch.randn(1, NUM_INPUT_PLANES, 8, 8)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["board_state"],
        output_names=["policy", "value"],
        dynamic_axes={
            "board_state": {0: "batch_size"},
            "policy": {0: "batch_size"},
            "value": {0: "batch_size"},
        },
    )

    # Print model size
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Exported ONNX model to {output_path} ({size_mb:.2f} MB)")

    # Verify
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(output_path)
        result = session.run(None, {"board_state": dummy_input.numpy()})
        print(
            f"Verification passed — policy shape: {result[0].shape}, "
            f"value shape: {result[1].shape}"
        )
    except ImportError:
        print("Install onnxruntime to verify: pip install onnxruntime")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export AlphaZero model to ONNX")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint file")
    parser.add_argument(
        "-o", "--output", default="model.onnx", help="Output ONNX file path"
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()
    export_to_onnx(args.checkpoint, args.output, args.opset)
