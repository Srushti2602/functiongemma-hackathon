#!/usr/bin/env python3
"""PrivacyLens — CLI orchestrator for video privacy redaction.

Uses FunctionGemma tool-calling (via generate_hybrid) to plan the redaction
pipeline, then executes each tool call in sequence.

Usage:
    python redact.py video.mp4 -i "blur faces and redact emails"
    python redact.py video.mp4  # uses default instruction
"""

import sys, os
sys.path.insert(0, "cactus/python/src")
os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

import argparse
import json
from dataclasses import dataclass, field

from main import generate_hybrid
from tools import ALL_TOOLS
from video_utils import sample_frames, apply_redactions
from detection import detect_faces, ocr_and_detect_pii


@dataclass
class RedactionContext:
    """Shared state passed between pipeline steps."""
    video_path: str = ""
    output_path: str = ""
    frame_dir: str = ""
    frame_paths: list = field(default_factory=list)
    sample_fps: int = 1
    face_detections: dict = field(default_factory=dict)
    pii_detections: dict = field(default_factory=dict)


# Pipeline step -> prerequisite steps
PREREQUISITES = {
    "detect_faces": ["sample_frames"],
    "ocr_and_detect_pii": ["sample_frames"],
    "apply_redactions": ["sample_frames"],  # at minimum; faces/pii added dynamically
}

# Full default pipeline
DEFAULT_PIPELINE = ["sample_frames", "detect_faces", "ocr_and_detect_pii", "apply_redactions"]


def execute_tool(tool_name, args, context):
    """Execute a single tool call and update the shared context."""
    print(f"\n>> Executing: {tool_name}({json.dumps(args, indent=2)})")

    if tool_name == "sample_frames":
        video_path = args.get("video_path", context.video_path)
        fps = int(args.get("fps", 1))
        context.sample_fps = fps
        context.frame_paths = sample_frames(video_path, fps)
        if context.frame_paths:
            context.frame_dir = os.path.dirname(context.frame_paths[0])

    elif tool_name == "detect_faces":
        if not context.frame_paths:
            print("  [auto] Running sample_frames first...")
            context.frame_paths = sample_frames(context.video_path, context.sample_fps)
            context.frame_dir = os.path.dirname(context.frame_paths[0])
        context.face_detections = detect_faces(context.frame_paths)

    elif tool_name == "ocr_and_detect_pii":
        if not context.frame_paths:
            print("  [auto] Running sample_frames first...")
            context.frame_paths = sample_frames(context.video_path, context.sample_fps)
            context.frame_dir = os.path.dirname(context.frame_paths[0])
        context.pii_detections = ocr_and_detect_pii(context.frame_paths)

    elif tool_name == "apply_redactions":
        if not context.frame_paths:
            print("  [auto] Running full detection pipeline first...")
            context.frame_paths = sample_frames(context.video_path, context.sample_fps)
            context.frame_dir = os.path.dirname(context.frame_paths[0])
        if not context.face_detections and not context.pii_detections:
            print("  [auto] Running face detection...")
            context.face_detections = detect_faces(context.frame_paths)
            print("  [auto] Running PII detection...")
            context.pii_detections = ocr_and_detect_pii(context.frame_paths)
        # Always use context paths (model may hallucinate paths)
        output = context.output_path or None
        apply_redactions(context.video_path, context, output)

    else:
        print(f"  Unknown tool: {tool_name}")


def ensure_complete_pipeline(tool_calls, instruction):
    """Add missing prerequisite steps if the model returns an incomplete pipeline."""
    called_names = [c["name"] for c in tool_calls]

    # Always need sample_frames at the beginning
    if "sample_frames" not in called_names:
        tool_calls.insert(0, {"name": "sample_frames", "arguments": {}})

    # Always need apply_redactions at the end
    if "apply_redactions" not in called_names:
        tool_calls.append({"name": "apply_redactions", "arguments": {}})

    # If instruction mentions faces/blur and detect_faces missing, add it
    instruction_lower = instruction.lower()
    if ("face" in instruction_lower or "blur" in instruction_lower) and "detect_faces" not in called_names:
        idx = next((i for i, c in enumerate(tool_calls) if c["name"] == "apply_redactions"), len(tool_calls))
        tool_calls.insert(idx, {"name": "detect_faces", "arguments": {}})

    # If instruction mentions email/pii/text/redact and ocr missing, add it
    if any(kw in instruction_lower for kw in ("email", "pii", "text", "phone", "ssn", "redact")) \
            and "ocr_and_detect_pii" not in called_names:
        idx = next((i for i, c in enumerate(tool_calls) if c["name"] == "apply_redactions"), len(tool_calls))
        tool_calls.insert(idx, {"name": "ocr_and_detect_pii", "arguments": {}})

    return tool_calls


def run_redaction(video_path, instruction="blur all faces and redact any PII text"):
    """Main orchestrator: plan via generate_hybrid, then execute."""
    print(f"PrivacyLens - Video Privacy Redaction")
    print(f"=" * 50)
    print(f"Video: {video_path}")
    print(f"Instruction: {instruction}")
    print(f"=" * 50)

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    context = RedactionContext(video_path=video_path)

    # Ask FunctionGemma to plan the pipeline
    messages = [{"role": "user", "content": instruction}]
    print("\n[1/2] Planning redaction pipeline via FunctionGemma...")
    result = generate_hybrid(messages, ALL_TOOLS)

    tool_calls = result.get("function_calls", [])
    source = result.get("source", "unknown")
    print(f"  Source: {source}")
    print(f"  Planned steps: {[c['name'] for c in tool_calls]}")

    # Ensure complete pipeline with smart fallback
    tool_calls = ensure_complete_pipeline(tool_calls, instruction)
    print(f"  Final pipeline: {[c['name'] for c in tool_calls]}")

    # Execute each step
    print(f"\n[2/2] Executing redaction pipeline...")
    for call in tool_calls:
        execute_tool(call["name"], call.get("arguments", {}), context)

    print(f"\nDone! Redacted video saved.")
    return context


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PrivacyLens - Video Privacy Redaction")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("-i", "--instruction", default="blur all faces and redact any PII text",
                        help="Natural language instruction for what to redact")
    parser.add_argument("-o", "--output", default=None, help="Output video path")
    args = parser.parse_args()

    ctx = run_redaction(args.video, args.instruction)
