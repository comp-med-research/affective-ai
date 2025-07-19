import pandas as pd
from pathlib import Path
from glob import glob


def generate_path_map(folder: Path, modality: str, is_video=False) -> dict:
    """Creates a UID → file path mapping for one modality.
    If is_video=True, it searches the raw MELD structure.
    """
    if is_video:
        files = glob(str(folder / "*.mp4"))  # e.g., MELD.Raw/train_splits/*.mp4
  
    else:
        files = glob(str(folder / modality / "*"))

    path_map = {}
    for path in files:
        name = Path(path).stem  # e.g., dia1_utt3
        if "dia" in name and "utt" in name:
            dialogue, utterance = name.replace("dia", "").split("_utt")
            uid = f"{dialogue}_{utterance}"
            path_map[uid] = path
    return path_map


def build_split_dataframe(split: str) -> pd.DataFrame:
    """Build merged DataFrame for train/dev/test split."""
    print(f"Processing split: {split}")

    # Text + labels (CSV)
    text_csv = Path(f"../data/processed/{split}/text/{split}_sent_emo.csv")
    if not text_csv.exists():
        raise FileNotFoundError(f"Missing: {text_csv}")
    
    df = pd.read_csv(text_csv)
    df["uid"] = df["Dialogue_ID"].astype(str) + "_" + df["Utterance_ID"].astype(str)

    # Paths for processed modalities
    processed_base = Path(f"../data/processed/{split}")
    audio_map = generate_path_map(processed_base, "audio")
    image_map = generate_path_map(processed_base, "images")

    # Path for raw video (unchanged MELD.Raw structure)
    meld_raw_video_dir = Path("../data/raw/MELD.Raw")
    video_map = generate_path_map(meld_raw_video_dir / split, modality=None, is_video=True)

    # Map paths to dataframe
    df["audio_path"] = df["uid"].map(audio_map)
    df["image_path"] = df["uid"].map(image_map)
    df["video_path"] = df["uid"].map(video_map)

    # Save merged CSV
    output_path = Path(f"../data/merged/{split}_merged.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved merged table: {output_path}")

    return df


if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        try:
            build_split_dataframe(split)
        except Exception as e:
            print(f"❌ Error with {split}: {e}")

