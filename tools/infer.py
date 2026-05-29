import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run I2M inference on one molecule image and print the predicted SMILES."
    )
    parser.add_argument("image", type=Path, help="Path to the input molecule image.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/I2M_R4.onnx"),
        help="Path to the ONNX checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/moldetr/moldetr_r50vd_6x_coco.yml"),
        help="Path to the model config.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/inference"),
        help="Working directory used to stage temporary inference inputs and outputs.",
    )
    parser.add_argument(
        "--dataset-name",
        default="custom_infer",
        help="Temporary dataset name passed to evaluate/eval_model.py.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep staged CSV / image / output files after inference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    image_path = args.image.resolve()
    weights_path = (repo_root / args.weights).resolve() if not args.weights.is_absolute() else args.weights.resolve()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    data_root = (repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root.resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    dataset_name = args.dataset_name
    dataset_dir = data_root / dataset_name
    csv_path = data_root / f"{dataset_name}.csv"
    output_csv = data_root / f"{dataset_name}_OUTPUTwithOCR.csv"
    none_mol_csv = data_root / f"{dataset_name}_none_mol_images.csv"
    summary_txt = Path(f"{output_csv}.I2Msummary.txt")

    data_root.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    staged_image = dataset_dir / image_path.name
    shutil.copy2(image_path, staged_image)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "SMILES"])
        writer.writeheader()
        writer.writerow({"file_path": f"{dataset_name}/{staged_image.name}", "SMILES": ""})

    if output_csv.exists():
        output_csv.unlink()
    if none_mol_csv.exists():
        none_mol_csv.unlink()

    cmd = [
        sys.executable,
        str(repo_root / "evaluate" / "eval_model.py"),
        "--config",
        str(config_path),
        "--resume",
        str(weights_path),
        "--dataname",
        dataset_name,
        "--number",
        "1",
        "--data-root",
        str(data_root),
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)

    if none_mol_csv.exists():
        print("No molecule structure could be reconstructed from the input image.", file=sys.stderr)
        return 2

    if not output_csv.exists():
        raise RuntimeError(f"Inference finished but no output CSV was created: {output_csv}")

    with output_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"Inference finished but produced an empty CSV: {output_csv}")

    result = rows[0]
    smiles = result.get("SMILESexp") or result.get("SMILESpre") or ""
    if not smiles:
        raise RuntimeError(f"Inference finished but no SMILES string was found in: {output_csv}")

    print(smiles)

    if not args.keep_artifacts:
        if output_csv.exists():
            output_csv.unlink()
        if none_mol_csv.exists():
            none_mol_csv.unlink()
        if summary_txt.exists():
            summary_txt.unlink()
        if staged_image.exists():
            staged_image.unlink()
        if dataset_dir.exists():
            try:
                dataset_dir.rmdir()
            except OSError:
                pass
        if csv_path.exists():
            csv_path.unlink()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
