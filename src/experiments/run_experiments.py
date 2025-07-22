import argparse
import yaml
import pandas as pd
from pathlib import Path
from ..utils.parsing import parse_emotion_response
import importlib
from tqdm import tqdm

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_inputs(row, modality: str):
    text = row.get("Utterance")
    image_path = row.get("image_path")
    audio_path = row.get("audio_path")

    return {
        "text": text if "text" in modality else None,
        "image_path": image_path if "image" in modality else None,
        "audio_path": audio_path if "audio" in modality else None,
    }

def main(config_path: str):
    config = load_config(config_path)

    model_name = config["model"]
    modality = config["modality"]  # e.g. "text", "image+audio"
    split = config.get("split", "test")
    output_file = config["output_file"]
    task = config.get("task", "emotion_classification")

    # Load model module dynamically
    model_module = importlib.import_module(f"src.models.{model_name}")

    # Load dataset
    merged_path = Path(f"data/merged/{split}_merged.csv")
    df = pd.read_csv(merged_path)

    predictions = []
    print(f"Running inference with {model_name} on modality: {modality} ({len(df)} examples)")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        inputs = build_inputs(row, modality)
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
