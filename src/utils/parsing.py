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
            # More robust fallback: search for valid emotions in the response
            response_lower = response.lower()
            found_emotion = None
            for valid_emotion in VALID_EMOTIONS:
                if valid_emotion in response_lower:
                    found_emotion = valid_emotion
                    break
            
            if found_emotion:
                emotion = found_emotion
                rationale = response.strip()
            else:
                # Last resort: assume first word is emotion
                emotion = response.strip().split()[0] if response.strip() else "unknown"
                rationale = response.strip()

        emotion_clean = emotion.strip().lower()
        if emotion_clean not in VALID_EMOTIONS:
            emotion = "unknown"    
    except Exception as e:
        emotion = "unknown"
        rationale = f"Parsing error: {e}"

    return emotion, rationale
