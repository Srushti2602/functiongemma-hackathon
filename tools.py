"""PrivacyLens — Tool definitions for FunctionGemma tool-calling."""

# ===== Video pipeline tools (original) =====

TOOL_SAMPLE_FRAMES = {
    "name": "sample_frames",
    "description": "Extract frames from a video file at a specified frame rate for analysis",
    "parameters": {
        "type": "object",
        "properties": {
            "video_path": {"type": "string", "description": "Path to the input video file"},
            "fps": {"type": "integer", "description": "Frames per second to extract"},
        },
        "required": ["video_path"],
    },
}

TOOL_DETECT_FACES = {
    "name": "detect_faces",
    "description": "Detect human faces in extracted video frames and return bounding box coordinates",
    "parameters": {
        "type": "object",
        "properties": {
            "frame_dir": {"type": "string", "description": "Directory containing extracted frame images"},
        },
        "required": ["frame_dir"],
    },
}

TOOL_OCR_AND_DETECT_PII = {
    "name": "ocr_and_detect_pii",
    "description": "Run OCR on video frames to extract text and detect personally identifiable information like emails, phone numbers, and addresses",
    "parameters": {
        "type": "object",
        "properties": {
            "frame_dir": {"type": "string", "description": "Directory containing extracted frame images"},
        },
        "required": ["frame_dir"],
    },
}

TOOL_APPLY_REDACTIONS = {
    "name": "apply_redactions",
    "description": "Apply privacy redactions to a video by blurring faces and covering PII text regions",
    "parameters": {
        "type": "object",
        "properties": {
            "video_path": {"type": "string", "description": "Path to the original input video file"},
            "output_path": {"type": "string", "description": "Path for the redacted output video file"},
        },
        "required": ["video_path"],
    },
}


# ===== FunctionGemma PII analysis tools (text-based) =====

TOOL_TRANSCRIBE = {
    "name": "transcribe_audio",
    "description": "Transcribe audio from a video or audio file into text with timestamps",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the video or audio file"},
        },
        "required": ["file_path"],
    },
}

TOOL_DETECT_PII_TEXT = {
    "name": "detect_pii_in_text",
    "description": "Scan text for personally identifiable information such as names, phone numbers, emails, addresses, and credit card numbers",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The text to scan for PII"},
        },
        "required": ["text"],
    },
}

TOOL_CLASSIFY_RISK = {
    "name": "classify_risk",
    "description": "Classify the privacy risk level of detected PII as low, medium, or high based on sensitivity",
    "parameters": {
        "type": "object",
        "properties": {
            "pii_count": {"type": "integer", "description": "Number of PII items found"},
            "pii_types": {"type": "string", "description": "Comma-separated list of PII types found"},
        },
        "required": ["pii_count", "pii_types"],
    },
}

TOOL_GENERATE_REPORT = {
    "name": "generate_privacy_report",
    "description": "Generate a structured privacy audit report summarizing all PII findings and recommended actions",
    "parameters": {
        "type": "object",
        "properties": {
            "media_type": {"type": "string", "description": "Type of media analyzed: video, audio, or image"},
            "findings_summary": {"type": "string", "description": "Summary of all PII findings"},
        },
        "required": ["media_type", "findings_summary"],
    },
}


# ===== Registries =====

ALL_TOOLS = [
    TOOL_SAMPLE_FRAMES,
    TOOL_DETECT_FACES,
    TOOL_OCR_AND_DETECT_PII,
    TOOL_APPLY_REDACTIONS,
]

# Tools for FunctionGemma-powered audio PII pipeline
AUDIO_PII_TOOLS = [
    TOOL_TRANSCRIBE,
    TOOL_DETECT_PII_TEXT,
    TOOL_CLASSIFY_RISK,
    TOOL_GENERATE_REPORT,
]

TOOL_REGISTRY = {t["name"]: t for t in ALL_TOOLS + AUDIO_PII_TOOLS}
