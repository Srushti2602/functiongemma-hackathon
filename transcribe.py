"""PrivacyLens — Audio transcription using Cactus Whisper (100% on-device)."""

import os
import sys
import json
import subprocess

sys.path.insert(0, "cactus/python/src")
os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

# Whisper model path — downloaded via `cactus download openai/whisper-small`
WHISPER_WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "cactus", "weights", "whisper-small"
)

_whisper_model = None


def _get_whisper_model():
    """Get or initialize the cached Cactus Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        from cactus import cactus_init
        path = os.path.abspath(WHISPER_WEIGHTS)
        print(f"Loading Cactus Whisper model from {path}...")
        _whisper_model = cactus_init(path)
        print("Cactus Whisper model loaded.")
    return _whisper_model


def extract_audio(video_path):
    """Extract audio from video to a temporary WAV file using ffmpeg."""
    audio_path = os.path.join("/tmp", "privacylens_audio",
                              os.path.basename(video_path) + ".wav")
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    if os.path.exists(audio_path):
        os.remove(audio_path)

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-y", audio_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr.decode()[:200]}")
            return None
    except FileNotFoundError:
        print("ffmpeg not found — trying direct audio load")
        return video_path
    except subprocess.TimeoutExpired:
        print("ffmpeg timed out")
        return None

    if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
        return audio_path
    return None


def transcribe_audio(video_path):
    """Transcribe video/audio to timestamped segments using Cactus Whisper.

    Returns:
        list[dict]: [{start, end, text}, ...] sorted by time.
    """
    from cactus import cactus_transcribe

    print(f"Extracting audio from {video_path}...")
    audio_path = extract_audio(video_path)
    if audio_path is None:
        print("Could not extract audio")
        return []

    model = _get_whisper_model()
    prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

    print("Transcribing with Cactus Whisper...")
    raw = cactus_transcribe(model, audio_path, prompt=prompt)
    print(f"Cactus Whisper raw output: {raw[:300]}...")

    # Parse the JSON response
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # If not JSON, treat the raw string as full transcript text
        if raw.strip():
            return [{"start": 0.0, "end": 0.0, "text": raw.strip()}]
        return []

    # If response has a "response" key with text
    text = data.get("response", "")
    if not text and isinstance(data, dict):
        text = data.get("text", "")

    if not text:
        print("No transcript text from Cactus Whisper")
        return []

    # Split into sentence-level segments (cactus doesn't provide timestamps natively)
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    segments = []
    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if sent:
            segments.append({
                "start": round(i * 3.0, 1),  # approximate timestamps
                "end": round((i + 1) * 3.0, 1),
                "text": sent,
            })

    print(f"Transcribed {len(segments)} segments via Cactus Whisper")
    return segments


def segments_to_full_text(segments):
    """Join all segments into one string for PII analysis."""
    return " ".join(s["text"] for s in segments)


def format_timestamp(seconds):
    """Format seconds as MM:SS."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"
