import os
import subprocess
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
VIDEO_DIRS = {
    "train": "../data/raw/MELD.Raw/train",
    "dev": "../data/raw/MELD.Raw/dev",
    "test": "../data/raw/MELD.Raw/test"
}


# === FUNCTION ===
def extract_from_video(video_path: Path, audio_path: Path, img_path: Path):
    # Extract audio (.wav)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(audio_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Extract 1 center frame as .jpg
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", "select=eq(n\\,0)", "-vframes", "1", str(img_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# === MAIN ===
for split, vid_dir in VIDEO_DIRS.items():
    print(f"Processing {split} split...")
    vid_dir = Path(vid_dir)
    files = sorted([f for f in vid_dir.glob("*.mp4")])

    AUDIO_OUT_DIR = Path(f"../data/processed/{split}/audio")
    IMG_OUT_DIR = Path(f"../data/processed/{split}/images")

    AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_OUT_DIR.mkdir(parents=True, exist_ok=True)

    for video_file in tqdm(files, desc=f"{split}"):
        base = video_file.stem  # e.g., dia123_utt4
        audio_out = AUDIO_OUT_DIR / f"{base}.wav"
        img_out = IMG_OUT_DIR / f"{base}.jpg"

        if not audio_out.exists() or not img_out.exists():
            extract_from_video(video_file, audio_out, img_out)
