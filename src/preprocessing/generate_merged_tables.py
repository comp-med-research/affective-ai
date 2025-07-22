import pandas as pd
from pathlib import Path
from glob import glob


def generate_path_map(folder: Path, modality: str = None, is_video=False) -> dict:
    """Creates a UID → absolute file path mapping for one modality."""
    if is_video:
        files = glob(str(folder / "*.mp4"))
    else:
        files = glob(str(folder / modality / "*"))

    path_map = {}
    for path in files:
        name = Path(path).stem
        if "dia" in name and "utt" in name:
            dialogue, utterance = name.replace("dia", "").split("_utt")
            uid = f"{dialogue}_{utterance}"
            path_map[uid] = str(Path(path).resolve())  # store absolute path
    return path_map


def build_split_dataframe(split: str) -> pd.DataFrame:
    """Build merged DataFrame for train/dev/test split."""
    print(f"Processing split: {split}")

    # Script base directory
    BASE_DIR = Path(__file__).resolve().parent.parent.parent  # assumes script is in src/
    processed_base = BASE_DIR / "data" / "processed" / split
    text_csv = processed_base / "text" / f"{split}_sent_emo.csv"
    if not text_csv.exists():
        raise FileNotFoundError(f"Missing: {text_csv}")
    
    df = pd.read_csv(text_csv)
    df["uid"] = df["Dialogue_ID"].astype(str) + "_" + df["Utterance_ID"].astype(str)

    # Generate absolute path maps
    audio_map = generate_path_map(processed_base, modality="audio")
    image_map = generate_path_map(processed_base, modality="images")
    video_base = BASE_DIR / "data" / "raw" / "MELD.Raw" / split
    video_map = generate_path_map(video_base, is_video=True)

    # Map to dataframe
    df["audio_path"] = df["uid"].map(audio_map)
    df["image_path"] = df["uid"].map(image_map)
    df["video_path"] = df["uid"].map(video_map)

    # Save merged CSV with absolute paths
    merged_dir = BASE_DIR / "data" / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    output_path = merged_dir / f"{split}_merged.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Saved merged table: {output_path}")

    return df


if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        try:
            build_split_dataframe(split)
        except Exception as e:
            print(f"❌ Error with {split}: {e}")


