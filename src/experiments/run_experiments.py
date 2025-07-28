import argparse
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import importlib

from src.utils.parsing import parse_emotion_response


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_inputs(row, modalities: list[str]):
    """Prepares input dict for inference based on selected modalities."""
    inputs = {}

    if "text" in modalities:
        inputs["text"] = row.get("Utterance")
    if "audio" in modalities:
        inputs["audio_path"] = row.get("audio_path")
    if "image" in modalities:
        inputs["image_path"] = row.get("image_path")
    if "video" in modalities:
        inputs["video_path"] = row.get("video_path")

    return inputs


def main(config_path: str):
    config = load_config(config_path)

    model_name = config["model"]
    raw_modalities = config["modalities"]  # Can be "text", or "text+audio"
    split = config.get("split", "test")
    output_file = config["output_file"]
    task = config.get("task", "emotion_classification")

    # Parse modalities into a list
    if isinstance(raw_modalities, str):
        modalities = [m.strip().lower() for m in raw_modalities.split("+")]
    else:
        modalities = [m.lower() for m in raw_modalities]

    # Load model module dynamically from src/models
    model_module = importlib.import_module(f"src.models.{model_name}")

    # Load dataset
    merged_path = Path(f"data/merged/{split}_merged.csv")
    if not merged_path.exists():
        raise FileNotFoundError(f"Missing dataset: {merged_path}")
    df = pd.read_csv(merged_path)

    predictions = []
    print(f"🧪 Running inference with {model_name} on modalities: {modalities} ({len(df)} examples)")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        inputs = build_inputs(row, modalities)
        response = model_module.infer(**inputs, task=task)
        emotion, rationale = parse_emotion_response(response)
        predictions.append((emotion, rationale))

    df["prediction"] = [p[0] for p in predictions]
    df["rationale"] = [p[1] for p in predictions]

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"✅ Saved predictions to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
