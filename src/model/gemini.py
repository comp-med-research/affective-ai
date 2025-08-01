import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from typing import Optional, Union
from pathlib import Path
import mimetypes
from PIL import Image

# Load your API key from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

MODEL = "gemini-2.0-flash-001"

client = genai.Client(
    http_options={"api_version": "v1alpha"},
    api_key=GOOGLE_API_KEY
)

def load_file(path: Union[str, Path]) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def get_mime_type(file_path: Union[str, Path]) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"

def build_multimodal_input(text: Optional[str], image_path: Optional[str] = None, audio_path: Optional[str] = None, video_path: Optional[str] = None):
    """Creates a multimodal input payload for Gemini."""
    parts = []

    if text:
        parts.append(text)

    if image_path:
        image = Image.open(image_path)
        parts.append(image)

    if audio_path:
        with open(audio_path, 'rb') as f:
            audio = f.read()
        parts.append(types.Part.from_bytes(
            data=audio,
            mime_type='audio/wav'
        ))
        
    if video_path:
        with open(video_path, 'rb') as f:
            video = f.read()
        parts.append(types.Part(
                inline_data=types.Blob(
                    data=video,
                    mime_type='video/mp4'
                )
        ))


    return parts

def infer(text: Optional[str] = None, image_path: Optional[str] = None, audio_path: Optional[str] = None, video_path: Optional[str] = None, task: str = "emotion_classification") -> str:
    """
    Run inference using Gemini on any combo of text/image/audio.
    Returns model's response text.
    """
    try:
        prompt_prefix = """
        Classify the speaker's emotion using **only one** of the following options: 
        anger, disgust, sadness, joy, neutral, surprise, fear.
        Respond in this format exactly:
        Emotion: <one-word label from the list above>
        Rationale: <brief explanation>
        """
        prompt_map = {
            "emotion_classification": prompt_prefix,
        }
        prompt = prompt_map[task]

        content = build_multimodal_input(text=prompt + "\n" + (text or ""), image_path=image_path, audio_path=audio_path, video_path=video_path)
        response = client.models.generate_content(
            model=MODEL,
            contents=content
        )
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini Error] {e}")
        return "error"
