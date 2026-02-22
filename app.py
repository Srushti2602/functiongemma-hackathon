#!/usr/bin/env python3
"""PrivacyLens — Flask web server for video/image privacy redaction.

Runs 100% offline using MediaPipe (faces) + EasyOCR (PII text).
No cloud API calls needed.
"""

import sys, os
sys.path.insert(0, "cactus/python/src")
os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

import json
import uuid
import base64
import threading
from flask import Flask, request, jsonify, send_file
from dataclasses import dataclass, field

import cv2
from video_utils import sample_frames, apply_redactions
from detection import detect_faces, cluster_faces, ocr_and_detect_pii
from transcribe import transcribe_audio, segments_to_full_text
from pii_analyzer import run_full_analysis, plan_analysis

app = Flask(__name__)

UPLOAD_DIR = os.path.join("/tmp", "privacylens_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

jobs = {}

# ===== Cactus Performance Tracking =====
cactus_stats = {
    "total_calls": 0,
    "on_device": 0,
    "cloud_fallback": 0,
    "total_time_ms": 0,
    "calls": [],  # last 20 calls: {source, time_ms, tools_planned, confidence}
}
cactus_lock = threading.Lock()

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}


@dataclass
class RedactionContext:
    video_path: str = ""
    output_path: str = ""
    frame_dir: str = ""
    frame_paths: list = field(default_factory=list)
    sample_fps: int = 2
    face_detections: dict = field(default_factory=dict)
    pii_detections: dict = field(default_factory=dict)
    blur_faces: bool = True
    face_clusters: dict = field(default_factory=dict)
    faces_to_blur: object = None


def run_scan(job_id):
    """Run detection pipeline (pauses for user face selection)."""
    job = jobs[job_id]
    file_path = job["file_path"]
    media_type = job["media_type"]

    try:
        # Step 1: Get frames
        job["status"] = "sampling"
        job["step"] = 1
        job["message"] = "Extracting frames..." if media_type == "video" else "Loading image..."

        if media_type == "video":
            frame_paths = sample_frames(file_path, fps=2)
        else:
            # For images, just use the image directly
            frame_paths = [file_path]

        job["frames_extracted"] = len(frame_paths)
        job["_frame_paths"] = frame_paths

        # Generate thumbnails
        thumbnails = []
        for fp in frame_paths[:8]:
            img = cv2.imread(fp)
            if img is not None:
                h, w = img.shape[:2]
                scale = 200 / max(h, w)
                thumb = cv2.resize(img, (int(w * scale), int(h * scale)))
                _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
                thumbnails.append(base64.b64encode(buf).decode())
        job["thumbnails"] = thumbnails

        # Step 2: Detect faces
        job["status"] = "detecting_faces"
        job["step"] = 2
        job["message"] = "Scanning for faces..."
        face_detections = detect_faces(frame_paths)
        job["_face_detections"] = face_detections

        total_faces = sum(len(v) for v in face_detections.values())
        job["faces_detected"] = total_faces

        clusters = cluster_faces(frame_paths, face_detections)
        job["_face_clusters"] = clusters
        job["unique_faces"] = len(clusters)

        face_options = []
        for cid, info in clusters.items():
            face_options.append({
                "id": cid,
                "thumbnail": info["thumbnail_b64"],
                "count": info["count"],
                "frames": len(info["frames"]),
            })
        job["face_options"] = face_options

        # Step 3: OCR + PII
        job["status"] = "detecting_pii"
        job["step"] = 3
        job["message"] = "Running OCR and detecting PII..."
        pii_detections = ocr_and_detect_pii(frame_paths)
        job["_pii_detections"] = pii_detections

        total_pii = sum(len(v) for v in pii_detections.values())
        job["pii_detected"] = total_pii

        pii_details = []
        for frame_key, bboxes in pii_detections.items():
            for bbox in bboxes:
                pii_details.append({"frame": frame_key, "bbox": list(bbox)})
        job["pii_details"] = pii_details

        # Step 3.25: Audio transcription + PII (for videos with audio)
        if media_type == "video":
            job["message"] = "Transcribing audio with Whisper..."
            try:
                segments = transcribe_audio(file_path)
                job["transcript_segments"] = len(segments)

                transcript_display = []
                for seg in segments:
                    from transcribe import format_timestamp
                    transcript_display.append({
                        "time": format_timestamp(seg["start"]),
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"],
                    })
                job["transcript"] = transcript_display

                from pii_analyzer import detect_pii_in_segments, classify_risk, mask_pii
                audio_findings = detect_pii_in_segments(segments)
                job["audio_findings"] = [
                    {**f, "masked": mask_pii(f["matched_text"], f["pii_type"])}
                    for f in audio_findings
                ]
                job["audio_pii_count"] = len(audio_findings)
                job["audio_pii_types"] = list(set(f["pii_type"] for f in audio_findings))
                job["audio_risk"] = classify_risk(audio_findings)
            except Exception as audio_err:
                print(f"Audio analysis skipped: {audio_err}")
                job["transcript"] = []
                job["audio_findings"] = []
                job["audio_pii_count"] = 0

        # Analysis complete — step must exceed total to mark all pipeline steps as "done"
        # Video: 4 steps (Extract → Faces → OCR → Audio), so step=5
        # Image: 3 steps (Load → Faces → OCR), so step=4
        job["status"] = "complete"
        job["step"] = 5 if media_type == "video" else 4
        job["message"] = "Analysis complete!"

    except Exception as e:
        job["status"] = "error"
        job["message"] = str(e)
        import traceback
        traceback.print_exc()


def run_redaction(job_id, faces_to_blur):
    """Apply redactions after user selects faces."""
    job = jobs[job_id]

    try:
        job["status"] = "redacting"
        job["step"] = 4
        job["message"] = "Applying redactions..."

        file_path = job["file_path"]
        media_type = job["media_type"]

        if media_type == "video":
            output_path = os.path.join(UPLOAD_DIR, job_id + "_redacted.mp4")
            context = RedactionContext(
                video_path=file_path,
                output_path=output_path,
                frame_paths=job.get("_frame_paths", []),
                sample_fps=2,
                face_detections=job.get("_face_detections", {}),
                pii_detections=job.get("_pii_detections", {}),
                blur_faces=True,
                face_clusters=job.get("_face_clusters", {}),
                faces_to_blur=faces_to_blur,
            )
            apply_redactions(file_path, context, output_path)
            job["output_path"] = output_path
            job["output_filename"] = os.path.basename(file_path).rsplit(".", 1)[0] + "_redacted.mp4"

        else:
            # Image redaction
            ext = os.path.splitext(file_path)[1]
            output_path = os.path.join(UPLOAD_DIR, job_id + "_redacted" + ext)
            redact_image(file_path, output_path, job, faces_to_blur)
            job["output_path"] = output_path
            job["output_filename"] = os.path.basename(file_path).rsplit(".", 1)[0] + "_redacted" + ext

        job["status"] = "complete"
        job["step"] = 5
        job["message"] = "Redaction complete!"

    except Exception as e:
        job["status"] = "error"
        job["message"] = str(e)
        import traceback
        traceback.print_exc()


def redact_image(input_path, output_path, job, faces_to_blur):
    """Apply face blur + PII blackout to a single image (Haar cascade)."""
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Cannot read image: " + input_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (fx, fy, fw, fh) in faces:
        roi = img[fy:fy+fh, fx:fx+fw]
        if roi.size > 0:
            blurred = cv2.GaussianBlur(roi, (99, 99), 30)
            img[fy:fy+fh, fx:fx+fw] = blurred

    # PII blackout
    frame_key = os.path.basename(input_path)
    pii_dets = job.get("_pii_detections", {})
    for (px, py, pw, ph) in pii_dets.get(frame_key, []):
        cv2.rectangle(img, (int(px), int(py)), (int(px+pw), int(py+ph)), (0, 0, 0), -1)

    cv2.imwrite(output_path, img)
    print(f"Redacted image saved to {output_path}")


@app.route("/")
def index():
    from flask import make_response
    with open("templates/index.html", "r") as f:
        html = f.read()
    response = make_response(html)
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/api/analyze_text", methods=["POST"])
def analyze_text():
    """Live PII analysis on text chunks (for real-time mic feature).
    Uses FunctionGemma to classify risk after regex detection.
    """
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"findings": [], "risk": {"level": "none"}})

    from pii_analyzer import detect_pii_in_segments, classify_risk
    segments = [{"start": 0, "end": 0, "text": text}]
    findings = detect_pii_in_segments(segments)

    from pii_analyzer import mask_pii
    masked = []
    for f in findings:
        masked.append({
            "pii_type": f["pii_type"],
            "matched_text": f["matched_text"],
            "masked": mask_pii(f["matched_text"], f["pii_type"]),
        })

    risk = classify_risk(findings) if findings else {"level": "none", "reason": ""}

    return jsonify({
        "findings": masked,
        "risk": risk,
        "count": len(findings),
    })


@app.route("/api/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["video"]
    instruction = request.form.get("instruction", "blur all faces and redact any PII text")

    job_id = str(uuid.uuid4())[:8]
    file_path = os.path.join(UPLOAD_DIR, job_id + "_" + file.filename)
    file.save(file_path)

    ext = os.path.splitext(file.filename)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        media_type = "image"
        img = cv2.imread(file_path)
        duration = 0
        width = img.shape[1] if img is not None else 0
        height = img.shape[0] if img is not None else 0
        fps = 0
    else:
        media_type = "video"
        cap = cv2.VideoCapture(file_path)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "step": 0,
        "message": "Queued for processing...",
        "filename": file.filename,
        "instruction": instruction,
        "file_path": file_path,
        "media_type": media_type,
        "duration": round(duration, 1),
        "resolution": f"{width}x{height}",
        "fps": round(fps, 1) if fps else 0,
        "frames_extracted": 0,
        "faces_detected": 0,
        "pii_detected": 0,
        "thumbnails": [],
        "face_options": [],
        "pii_details": [],
        "output_path": None,
        "output_filename": None,
        "transcript": [],
        "transcript_segments": 0,
        "audio_findings": [],
        "audio_pii_count": 0,
        "audio_pii_types": [],
        "audio_risk": {},
    }

    thread = threading.Thread(target=run_scan, args=(job_id,))
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/apply/<job_id>", methods=["POST"])
def apply_route(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    data = request.get_json()
    selected_ids = data.get("faces_to_blur", [])

    if selected_ids is None or selected_ids == "all":
        faces_to_blur = None
    else:
        faces_to_blur = set(int(x) for x in selected_ids)

    thread = threading.Thread(target=run_redaction, args=(job_id, faces_to_blur))
    thread.start()

    return jsonify({"status": "redacting"})


@app.route("/api/status/<job_id>")
def status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    job = dict(jobs[job_id])
    for key in list(job.keys()):
        if key.startswith("_"):
            del job[key]
    job.pop("file_path", None)
    job.pop("output_path", None)
    return jsonify(job)


@app.route("/api/download/<job_id>")
def download(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    job = jobs[job_id]
    if job["status"] != "complete" or not job.get("output_path"):
        return jsonify({"error": "Not ready"}), 400
    return send_file(
        job["output_path"],
        as_attachment=True,
        download_name=job.get("output_filename", "redacted"),
    )


# ============ Audio PII Analysis (FunctionGemma-powered) ============

AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'}

def run_audio_scan(job_id):
    """Run the FunctionGemma-powered audio PII analysis pipeline."""
    job = jobs[job_id]
    file_path = job["file_path"]
    instruction = job.get("instruction", "Scan for any personal information")

    try:
        # Step 1: FunctionGemma plans the analysis
        job["status"] = "planning"
        job["step"] = 1
        job["message"] = "FunctionGemma planning analysis..."

        plan = plan_analysis(instruction)
        job["fg_plan"] = plan["planned_tools"]
        job["fg_source"] = plan["source"]
        job["fg_time_ms"] = plan["time_ms"]
        track_cactus_call({"source": plan["source"], "total_time_ms": plan["time_ms"], "function_calls": plan.get("raw_calls", []), "confidence": 1.0})

        # Step 2: Transcribe audio
        job["status"] = "transcribing"
        job["step"] = 2
        job["message"] = "Transcribing audio with Whisper..."

        segments = transcribe_audio(file_path)
        job["_segments"] = segments
        job["transcript_segments"] = len(segments)

        transcript_text = segments_to_full_text(segments)
        # Store preview (first 500 chars)
        job["transcript_preview"] = transcript_text[:500] + ("..." if len(transcript_text) > 500 else "")
        job["transcript_full"] = transcript_text

        # Build transcript for display
        transcript_display = []
        for seg in segments:
            from transcribe import format_timestamp
            transcript_display.append({
                "time": format_timestamp(seg["start"]),
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            })
        job["transcript"] = transcript_display

        # Step 3: FunctionGemma detects PII
        job["status"] = "detecting_pii"
        job["step"] = 3
        job["message"] = "FunctionGemma analyzing transcript for PII..."

        analysis = run_full_analysis(segments, instruction)

        job["audio_findings"] = analysis["findings"]
        job["audio_risk"] = analysis["risk"]
        job["audio_pii_count"] = analysis["total_pii"]
        job["audio_pii_types"] = analysis["pii_types"]
        job["fg_risk_source"] = analysis["risk"].get("source", "unknown")

        # Step 4: Done
        job["status"] = "complete"
        job["step"] = 4
        job["message"] = "Audio PII analysis complete!"

    except Exception as e:
        job["status"] = "error"
        job["message"] = str(e)
        import traceback
        traceback.print_exc()


@app.route("/api/upload_audio", methods=["POST"])
def upload_audio():
    """Upload a video/audio file for audio-based PII analysis."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    instruction = request.form.get("instruction", "Scan audio for any personal information")

    job_id = str(uuid.uuid4())[:8]
    file_path = os.path.join(UPLOAD_DIR, job_id + "_" + file.filename)
    file.save(file_path)

    ext = os.path.splitext(file.filename)[1].lower()

    # Get duration if video
    duration = 0
    if ext not in AUDIO_EXTENSIONS:
        try:
            cap = cv2.VideoCapture(file_path)
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
            cap.release()
        except Exception:
            pass

    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "step": 0,
        "message": "Queued for audio PII analysis...",
        "filename": file.filename,
        "instruction": instruction,
        "file_path": file_path,
        "media_type": "audio_pii",
        "duration": round(duration, 1),
        "transcript_segments": 0,
        "transcript_preview": "",
        "transcript_full": "",
        "transcript": [],
        "audio_findings": [],
        "audio_risk": {},
        "audio_pii_count": 0,
        "audio_pii_types": [],
        "fg_plan": [],
        "fg_source": "",
        "fg_time_ms": 0,
        "fg_risk_source": "",
    }

    thread = threading.Thread(target=run_audio_scan, args=(job_id,))
    thread.start()

    return jsonify({"job_id": job_id})


def track_cactus_call(result):
    """Track a FunctionGemma/Cactus call for the performance dashboard."""
    with cactus_lock:
        cactus_stats["total_calls"] += 1
        source = result.get("source", "unknown")
        if "on-device" in source:
            cactus_stats["on_device"] += 1
        else:
            cactus_stats["cloud_fallback"] += 1
        ms = result.get("total_time_ms", 0) or result.get("time_ms", 0)
        cactus_stats["total_time_ms"] += ms
        cactus_stats["calls"].append({
            "source": source,
            "time_ms": round(ms, 1),
            "confidence": result.get("confidence", 0),
            "tools": [c.get("name", "") for c in result.get("function_calls", [])],
            "timestamp": int(__import__("time").time() * 1000),
        })
        # Keep last 50
        if len(cactus_stats["calls"]) > 50:
            cactus_stats["calls"] = cactus_stats["calls"][-50:]


@app.route("/api/cactus_stats")
def get_cactus_stats():
    """Return Cactus/FunctionGemma performance stats for the dashboard."""
    from main import classify_query

    with cactus_lock:
        total = cactus_stats["total_calls"] or 1
        on_device = cactus_stats["on_device"]
        cloud = cactus_stats["cloud_fallback"]
        avg_ms = round(cactus_stats["total_time_ms"] / total, 1) if total > 0 else 0

        # Get cactus config params
        params = {
            "confidence_threshold": 0.2,
            "tool_rag_top_k": 0,
            "force_tools": True,
            "max_tokens": 128,
            "temperature": 0.01,
            "model": "FunctionGemma 270M (on-device)",
            "fallback": "Gemini (cloud)",
            "routing": "classify_query -> easy/medium/hard",
            "post_processing": "garbage detection + arg fixing",
        }

        return jsonify({
            "total_calls": cactus_stats["total_calls"],
            "on_device": on_device,
            "cloud_fallback": cloud,
            "on_device_ratio": round(on_device / total * 100, 1) if total > 0 else 0,
            "avg_time_ms": avg_ms,
            "total_time_ms": round(cactus_stats["total_time_ms"], 1),
            "recent_calls": cactus_stats["calls"][-20:],
            "params": params,
        })


@app.route("/api/cactus_demo", methods=["POST"])
def cactus_demo():
    """Run a live FunctionGemma demo call and return the raw result."""
    data = request.get_json()
    prompt = data.get("prompt", "What's the weather in San Francisco?")

    from main import generate_hybrid
    from tools import AUDIO_PII_TOOLS, ALL_TOOLS

    tool_set = data.get("tool_set", "audio")
    tools = AUDIO_PII_TOOLS if tool_set == "audio" else ALL_TOOLS

    messages = [{"role": "user", "content": prompt}]
    result = generate_hybrid(messages, tools)
    track_cactus_call(result)

    return jsonify({
        "prompt": prompt,
        "source": result.get("source", "unknown"),
        "function_calls": result.get("function_calls", []),
        "confidence": result.get("confidence", 0),
        "time_ms": round(result.get("total_time_ms", 0), 1),
        "tool_set": tool_set,
    })


if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    print("\n  PrivacyLens Web UI (100% Offline)")
    print("  http://localhost:5001")
    print("  Supports: Video + Images + Audio PII Analysis")
    print("  Powered by FunctionGemma (Cactus) + Whisper + MediaPipe\n")
    app.run(host="0.0.0.0", port=5001, debug=False)
