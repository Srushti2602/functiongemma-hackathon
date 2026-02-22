"""PrivacyLens — OpenCV frame extraction and redaction rendering."""

import os
import cv2
import numpy as np


def sample_frames(video_path, fps=1):
    """Extract frames from a video at the given FPS rate.

    Returns:
        list[str]: Paths to extracted frame images in output_dir.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(video_fps / fps))

    output_dir = os.path.join("/tmp", "privacylens_frames", os.path.basename(video_path))
    os.makedirs(output_dir, exist_ok=True)

    frame_paths = []
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            path = os.path.join(output_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(path, frame)
            frame_paths.append(path)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"Extracted {saved} frames to {output_dir}")
    return frame_paths


def apply_redactions(video_path, context, output_path=None):
    """Apply redactions to EVERY frame using real-time face detection.

    For consistent blur: runs face detection on every single frame (not just
    sampled ones). Only blurs faces whose cluster ID is in context.faces_to_blur.
    If faces_to_blur is None, blurs all faces.
    """
    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_redacted.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))

    # Set up real-time face detector for every frame
    import mediapipe as mp
    from detection import _get_face_model_path

    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=_get_face_model_path()),
        min_detection_confidence=0.5,
    )

    # Which faces to blur (by cluster index). None = blur all.
    faces_to_blur = getattr(context, 'faces_to_blur', None)
    face_clusters = getattr(context, 'face_clusters', None)

    # PII detection is still sample-based
    sample_fps = context.sample_fps or 1
    frame_interval = max(1, int(video_fps / sample_fps))

    frame_idx = 0
    sample_idx = 0
    total_blurred = 0

    with FaceDetector.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # --- Face blur on EVERY frame ---
            if getattr(context, 'blur_faces', True):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = detector.detect(mp_image)

                for det in results.detections:
                    bbox = det.bounding_box
                    fx, fy, fw, fh = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

                    should_blur = True
                    if faces_to_blur is not None and face_clusters is not None:
                        # Match this detection to a known cluster by center proximity
                        cx, cy = fx + fw // 2, fy + fh // 2
                        should_blur = _match_face_to_blur_list(
                            cx, cy, face_clusters, faces_to_blur, width, height
                        )

                    if should_blur:
                        x, y = max(0, int(fx)), max(0, int(fy))
                        w, h = int(fw), int(fh)
                        roi = frame[y:y+h, x:x+w]
                        if roi.size > 0:
                            blurred = cv2.GaussianBlur(roi, (99, 99), 30)
                            frame[y:y+h, x:x+w] = blurred
                            total_blurred += 1

            # --- PII redaction on sampled frames only ---
            if frame_idx % frame_interval == 0:
                frame_key = f"frame_{sample_idx:05d}.jpg"
                for (px, py, pw, ph) in context.pii_detections.get(frame_key, []):
                    px, py, pw, ph = int(px), int(py), int(pw), int(ph)
                    cv2.rectangle(frame, (px, py), (px+pw, py+ph), (0, 0, 0), -1)
                sample_idx += 1

            out.write(frame)
            frame_idx += 1

    cap.release()
    out.release()
    print(f"Redacted video saved to {output_path} ({total_blurred} face blurs applied across {frame_idx} frames)")
    return output_path


def _match_face_to_blur_list(cx, cy, face_clusters, faces_to_blur, frame_w, frame_h):
    """Check if a detected face center matches any cluster marked for blurring."""
    # Normalize coordinates
    ncx, ncy = cx / frame_w, cy / frame_h
    best_dist = float('inf')
    best_cluster = -1

    for cluster_id, cluster_info in face_clusters.items():
        # cluster_info has 'center_x', 'center_y' as normalized coords
        dist = ((ncx - cluster_info['center_x']) ** 2 + (ncy - cluster_info['center_y']) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_cluster = int(cluster_id)

    # If closest cluster is within reasonable distance AND in blur list
    if best_dist < 0.3 and best_cluster in faces_to_blur:
        return True
    return False
