# src/utils/parsing.py

VALID_EMOTIONS = {
    "anger", "disgust", "sadness", "joy", "neutral", "surprise", "fear"
}

def parse_emotion_response(response: str) -> tuple[str, str]:
    """
    Parses the model's response into (emotion, rationale).

    Expected format:
    Emotion: <one-word label>
    Rationale: <brief explanation>

    Returns:
        (emotion, rationale)
    """
    if not response:
        return "unknown", "No response"

    emotion = "unknown"
    rationale = "No rationale found"

    try:
        if "Emotion:" in response and "Rationale:" in response:
            emotion = response.split("Emotion:")[1].split("Rationale:")[0].strip()
            rationale = response.split("Rationale:")[1].strip()
        else:
            # fallback: assume first word is emotion
            emotion = response.strip().split()[0]
            rationale = response.strip()

        emotion_clean = emotion.strip().lower()
        if emotion_clean not in VALID_EMOTIONS:
            emotion = "unknown"    
    except Exception as e:
        rationale = f"Parsing error: {e}"

    return emotion, rationale
