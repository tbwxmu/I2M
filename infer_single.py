import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evaluate"))

from det_engine import RTDETRPostProcessor  # noqa: E402
from src.solver.utils import output_to_smiles  # noqa: E402


IDX_TO_LABELS = {
    0: "other",
    1: "C",
    2: "O",
    3: "N",
    4: "Cl",
    5: "Br",
    6: "S",
    7: "F",
    8: "B",
    9: "I",
    10: "P",
    11: "H",
    12: "Si",
    13: "single",
    14: "wdge",
    15: "dash",
    16: "=",
    17: "#",
    18: ":",
    19: "-4",
    20: "-2",
    21: "-1",
    22: "+1",
    23: "+2",
}
BOND_LABELS = [13, 14, 15, 16, 17, 18]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run direct single-image inference with I2M and print the predicted SMILES."
    )
    parser.add_argument("image", type=Path, help="Path to the input molecule image.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/I2M_R4.onnx"),
        help="Path to the ONNX checkpoint.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold applied before graph reconstruction.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution provider for ONNX Runtime.",
    )
    parser.add_argument(
        "--print-detections",
        action="store_true",
        help="Print the filtered detections before SMILES reconstruction.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_image_as_tensor(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    resized = image.resize((640, 640))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = np.transpose(array, (2, 0, 1))[None, ...]
    return tensor, width, height


def get_providers(device: str):
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def run_model(session: ort.InferenceSession, tensor: np.ndarray, width: int, height: int):
    inputs = session.get_inputs()
    if len(inputs) == 2:
        feed = {
            inputs[0].name: tensor,
            inputs[1].name: np.array([[width, height]], dtype=np.int64),
        }
    else:
        feed = {inputs[0].name: tensor}
    return session.run(None, feed)


def postprocess_outputs(outputs, width: int, height: int):
    postprocessor = RTDETRPostProcessor(
        classes_dict=IDX_TO_LABELS,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False,
    )
    ori_size = torch.tensor([[width, height]], dtype=torch.long)

    if len(outputs) == 1:
        output_tensor = torch.from_numpy(outputs[0])
        if output_tensor.shape[-1] <= 4:
            raise ValueError(f"Unexpected ONNX output shape: {tuple(output_tensor.shape)}")
        boxes = output_tensor[0, :, :4]
        logits = output_tensor[0, :, 4:]
        return postprocessor({"pred_logits": logits, "pred_boxes": boxes}, ori_size)[0]

    if len(outputs) == 2:
        return postprocessor(
            {
                "pred_logits": torch.from_numpy(outputs[0]),
                "pred_boxes": torch.from_numpy(outputs[1]),
            },
            ori_size,
        )[0]

    if len(outputs) == 3:
        arrays = list(outputs)
        boxes_idx = next((i for i, arr in enumerate(arrays) if arr.shape[-1] == 4), 0)
        remaining = [i for i in range(3) if i != boxes_idx]
        score_like = []
        label_like = []
        for idx in remaining:
            arr = arrays[idx]
            if np.issubdtype(arr.dtype, np.floating):
                score_like.append(idx)
            else:
                label_like.append(idx)
        scores_idx = score_like[0] if score_like else remaining[0]
        labels_idx = label_like[0] if label_like else remaining[1]

        boxes = arrays[boxes_idx][0] if arrays[boxes_idx].ndim == 3 else arrays[boxes_idx]
        scores = arrays[scores_idx][0] if arrays[scores_idx].ndim == 2 else arrays[scores_idx]
        labels = arrays[labels_idx][0] if arrays[labels_idx].ndim == 2 else arrays[labels_idx]
        return {
            "boxes": torch.from_numpy(boxes).float(),
            "scores": torch.from_numpy(scores).float(),
            "labels": torch.from_numpy(labels).long(),
        }

    raise ValueError(f"Unexpected number of ONNX outputs: {len(outputs)}")


def filter_result(result, threshold: float):
    keep = result["scores"] > threshold
    return {
        "boxes": result["boxes"][keep],
        "scores": result["scores"][keep],
        "labels": result["labels"][keep],
    }


def main():
    args = parse_args()
    image_path = args.image.resolve()
    weights_path = resolve_path(args.weights)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    tensor, width, height = load_image_as_tensor(image_path)
    session = ort.InferenceSession(str(weights_path), providers=get_providers(args.device))
    outputs = run_model(session, tensor, width, height)
    result = postprocess_outputs(outputs, width, height)
    filtered = filter_result(result, args.threshold)

    if filtered["boxes"].numel() == 0:
        raise RuntimeError("No detections survived the confidence threshold.")

    if args.print_detections:
        for label, score, box in zip(filtered["labels"], filtered["scores"], filtered["boxes"]):
            print(
                {
                    "label": IDX_TO_LABELS[int(label)],
                    "score": float(score),
                    "box": [float(x) for x in box],
                },
                file=sys.stderr,
            )

    _, smiles, _, _ = output_to_smiles(filtered, IDX_TO_LABELS, BOND_LABELS, result=None)
    if not smiles:
        raise RuntimeError("Failed to reconstruct a SMILES string from the detected graph.")

    print(smiles)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
