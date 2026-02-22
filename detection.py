"""PrivacyLens — Face detection (MediaPipe) and OCR/PII detection (EasyOCR)."""

import os
import re
import base64
import cv2
import numpy as np


# --- Location knowledge base (countries, major cities, US states) ---
COUNTRIES = {
    "afghanistan", "albania", "algeria", "argentina", "armenia", "australia",
    "austria", "azerbaijan", "bahamas", "bahrain", "bangladesh", "barbados",
    "belarus", "belgium", "belize", "bhutan", "bolivia", "bosnia", "botswana",
    "brazil", "brunei", "bulgaria", "cambodia", "cameroon", "canada", "chile",
    "china", "colombia", "congo", "costa rica", "croatia", "cuba", "cyprus",
    "czech republic", "denmark", "dominica", "ecuador", "egypt", "el salvador",
    "england", "estonia", "ethiopia", "fiji", "finland", "france", "gabon",
    "georgia", "germany", "ghana", "greece", "grenada", "guatemala", "guinea",
    "guyana", "haiti", "honduras", "hungary", "iceland", "india", "indonesia",
    "iran", "iraq", "ireland", "israel", "italy", "jamaica", "japan", "jordan",
    "kazakhstan", "kenya", "korea", "kuwait", "laos", "latvia", "lebanon",
    "libya", "lithuania", "luxembourg", "madagascar", "malaysia", "maldives",
    "mali", "malta", "mexico", "moldova", "monaco", "mongolia", "montenegro",
    "morocco", "mozambique", "myanmar", "namibia", "nepal", "netherlands",
    "new zealand", "nicaragua", "niger", "nigeria", "north korea", "norway",
    "oman", "pakistan", "palestine", "panama", "paraguay", "peru", "philippines",
    "poland", "portugal", "qatar", "romania", "russia", "rwanda", "saudi arabia",
    "scotland", "senegal", "serbia", "singapore", "slovakia", "slovenia",
    "somalia", "south africa", "south korea", "spain", "sri lanka", "sudan",
    "sweden", "switzerland", "syria", "taiwan", "tanzania", "thailand", "togo",
    "trinidad", "tunisia", "turkey", "uganda", "ukraine", "united arab emirates",
    "united kingdom", "united states", "uruguay", "uzbekistan", "venezuela",
    "vietnam", "wales", "yemen", "zambia", "zimbabwe",
    # Common abbreviations
    "uk", "us", "usa", "uae",
}

MAJOR_CITIES = {
    "new york", "los angeles", "chicago", "houston", "phoenix", "philadelphia",
    "san antonio", "san diego", "dallas", "san jose", "austin", "san francisco",
    "seattle", "denver", "boston", "nashville", "detroit", "portland", "memphis",
    "atlanta", "miami", "orlando", "tampa", "charlotte", "pittsburgh", "las vegas",
    "london", "paris", "berlin", "madrid", "rome", "amsterdam", "vienna",
    "brussels", "lisbon", "prague", "warsaw", "budapest", "dublin", "zurich",
    "geneva", "stockholm", "oslo", "copenhagen", "helsinki", "athens",
    "moscow", "istanbul", "barcelona", "munich", "hamburg", "milan",
    "tokyo", "osaka", "beijing", "shanghai", "hong kong", "singapore",
    "seoul", "taipei", "bangkok", "mumbai", "delhi", "bangalore", "chennai",
    "hyderabad", "kolkata", "dubai", "abu dhabi", "doha", "riyadh",
    "sydney", "melbourne", "brisbane", "perth", "auckland", "wellington",
    "toronto", "montreal", "vancouver", "calgary", "ottawa",
    "mexico city", "sao paulo", "rio de janeiro", "buenos aires", "lima",
    "bogota", "santiago", "cairo", "cape town", "johannesburg", "nairobi",
    "lagos", "casablanca", "marrakech",
}

US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
    "rhode island", "south carolina", "south dakota", "tennessee", "texas",
    "utah", "vermont", "virginia", "washington", "west virginia",
    "wisconsin", "wyoming",
}

ALL_LOCATIONS = COUNTRIES | MAJOR_CITIES | US_STATES

# Build a word-boundary regex for single-word locations (case-insensitive)
_single_word_locs = sorted(
    [loc for loc in ALL_LOCATIONS if ' ' not in loc and len(loc) > 2],
    key=len, reverse=True
)
_multi_word_locs = sorted(
    [loc for loc in ALL_LOCATIONS if ' ' in loc],
    key=len, reverse=True
)

# PII regex patterns
PII_PATTERNS = {
    "email": re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
    "phone": re.compile(
        r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'  # (555) 123-4567
        r'|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'                               # 555-123-4567
        r'|\b\d{10}\b'                                                      # 5551234567
        r'|\+\d{1,3}[-.\s]?\d{6,12}\b'                                     # +1 5551234567
    ),
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
    "license_plate": re.compile(
        r'\b[A-Z]{2,3}[-\s]?\d{1,4}[-\s]?[A-Z]{0,3}\b'
        r'|\b\d{1,4}[-\s]?[A-Z]{2,3}[-\s]?\d{0,4}\b'
        r'|\b[A-Z]{1,3}\d{1,2}[-\s]?[A-Z]{3}\b'
        r'|\b[A-Z]{2}\d{2}[-\s]?[A-Z]{2}[-\s]?\d{4}\b'
    ),
    "address": re.compile(
        r'\b\d+\s+(?:[A-Za-z]+\s+){0,3}'
        r'(?:Street|St|Avenue|Ave|Boulevard|Blvd|Drive|Dr|Road|Rd|Lane|Ln|Way|Court|Ct'
        r'|Place|Pl|Circle|Cir|Terrace|Ter|Trail|Trl|Parkway|Pkwy|Highway|Hwy|Square|Sq)\b',
        re.IGNORECASE
    ),
    "ip_address": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    "date_of_birth": re.compile(
        r'\b(?:born\s+(?:on\s+)?|dob[:\s]+|date of birth[:\s]+)'
        r'(?:\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})',
        re.IGNORECASE
    ),
    "url": re.compile(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'),
}

# Spoken phone number words → detect "five five five one two three four five six seven"
_SPOKEN_DIGITS = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    'oh': '0', 'o': '0',
}
SPOKEN_PHONE_RE = re.compile(
    r'\b(?:(?:zero|one|two|three|four|five|six|seven|eight|nine|oh|o)\s+){6,}\b'
    r'(?:zero|one|two|three|four|five|six|seven|eight|nine|oh|o)\b',
    re.IGNORECASE,
)

# Name context patterns for speech
NAME_CONTEXT_PATTERNS = [
    re.compile(r"(?:my name is|i'm|this is|call me|i am|named|name's)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.IGNORECASE),
    re.compile(r"(?:Mr|Mrs|Ms|Dr|Miss|Professor|Prof|Officer|Captain|Capt)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.IGNORECASE),
    re.compile(r"(?:his name is|her name is|their name is|known as|goes by|called)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.IGNORECASE),
    re.compile(r"(?:meet|introduce|introducing|presenting)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.IGNORECASE),
    re.compile(r"(?:contact|reach|email|call|text|message)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", re.IGNORECASE),
]

# Address context patterns for speech
ADDRESS_CONTEXT_PATTERNS = [
    re.compile(
        r'(?:live at|lives at|located at|address is|reside at|stay at|moved to|based at|office at)\s+'
        r'(\d+\s+[A-Z][a-zA-Z]+(?:\s+[A-Za-z]+){0,4})',
        re.IGNORECASE
    ),
    re.compile(
        r'(?:live (?:in|on)|lives (?:in|on)|located (?:in|on)|reside (?:in|on)|stay (?:in|on))\s+'
        r'([A-Z][a-zA-Z]+(?:\s+[A-Za-z]+){0,3})',
        re.IGNORECASE
    ),
]

# Common non-name words to filter out false positives
_COMMON_WORDS = {
    'the', 'and', 'but', 'for', 'are', 'not', 'you', 'all', 'can', 'had', 'her',
    'was', 'one', 'our', 'out', 'has', 'his', 'how', 'its', 'may', 'new', 'now',
    'old', 'see', 'way', 'who', 'did', 'get', 'let', 'say', 'she', 'too', 'use',
    'yes', 'just', 'have', 'will', 'been', 'this', 'that', 'with', 'from', 'they',
    'what', 'when', 'like', 'your', 'them', 'then', 'than', 'some', 'also', 'were',
    'very', 'much', 'into', 'over', 'such', 'here', 'take', 'well', 'really', 'about',
    'would', 'could', 'should', 'today', 'tomorrow', 'yesterday', 'beautiful', 'good',
    'great', 'nice', 'sure', 'please', 'thanks', 'thank', 'hello', 'sorry', 'right',
    'left', 'back', 'next', 'last', 'first', 'second', 'third', 'we', 'he', 'it',
    'be', 'to', 'of', 'in', 'so', 'no', 'do', 'if', 'or', 'an', 'on', 'is', 'at',
    'up', 'my', 'go', 'me', 'am', 'oh', 'ok', 'hi', 'hey', 'need', 'want', 'make',
    'know', 'think', 'come', 'look', 'work', 'help', 'talk', 'tell', 'give', 'live',
    'feel', 'find', 'long', 'down', 'after', 'before', 'while', 'where', 'there',
    'every', 'again', 'never', 'always', 'still', 'each', 'both', 'these', 'those',
    'being', 'same', 'other', 'which', 'their', 'only', 'more', 'most', 'people',
    'time', 'many', 'made', 'keep', 'going', 'start', 'end', 'point', 'thing',
    'enjoy', 'enjoying', 'beautiful', 'wonderful', 'amazing', 'awesome', 'okay',
    'maybe', 'actually', 'probably', 'definitely', 'however', 'though', 'already',
    'because', 'since', 'until', 'something', 'everything', 'nothing', 'anything',
    'someone', 'everyone', 'anyone', 'sometimes', 'place', 'home', 'house', 'room',
    'door', 'world', 'year', 'years', 'month', 'months', 'week', 'weeks', 'day',
    'days', 'morning', 'evening', 'night', 'afternoon',
    # Street/address words (avoid false name matches)
    'street', 'avenue', 'boulevard', 'drive', 'road', 'lane', 'way', 'court',
    'place', 'circle', 'terrace', 'trail', 'parkway', 'highway', 'square',
    'north', 'south', 'east', 'west', 'main', 'park', 'lake', 'river',
    'oak', 'elm', 'pine', 'maple', 'cedar', 'baker', 'king', 'queen',
    # Verbs (avoid false name matches from context patterns with IGNORECASE)
    'called', 'said', 'told', 'asked', 'went', 'came', 'left', 'got',
    'saw', 'heard', 'took', 'gave', 'put', 'ran', 'sat', 'stood',
    'loved', 'liked', 'lived', 'worked', 'moved', 'stayed', 'played',
    'called', 'visited', 'arrived', 'returned', 'noticed', 'mentioned',
    'recently', 'apparently', 'basically', 'certainly', 'clearly',
    'currently', 'excited', 'wonderful', 'fantastic', 'terrible', 'incredible',
    'absolutely', 'seriously', 'honestly', 'obviously', 'totally', 'completely',
    'finally', 'suddenly', 'quickly', 'slowly', 'carefully', 'exactly',
    'especially', 'particularly', 'generally', 'usually', 'often', 'once',
    'twice', 'tonight', 'tomorrow', 'waiting', 'looking', 'walking',
    'running', 'sitting', 'standing', 'talking', 'singing', 'reading',
    'writing', 'eating', 'drinking', 'sleeping', 'driving', 'flying',
}


def find_locations_in_text(text):
    """Find location names (countries, cities, states) in text.

    Returns list of (matched_text, start_index, end_index).
    """
    findings = []
    text_lower = text.lower()

    # Check multi-word locations first
    for loc in _multi_word_locs:
        idx = 0
        while True:
            pos = text_lower.find(loc, idx)
            if pos == -1:
                break
            end = pos + len(loc)
            # Check word boundaries
            before_ok = pos == 0 or not text_lower[pos - 1].isalpha()
            after_ok = end >= len(text_lower) or not text_lower[end].isalpha()
            if before_ok and after_ok:
                # Use original case from text
                findings.append((text[pos:end], pos, end))
            idx = pos + 1

    # Check single-word locations
    for loc in _single_word_locs:
        idx = 0
        while True:
            pos = text_lower.find(loc, idx)
            if pos == -1:
                break
            end = pos + len(loc)
            before_ok = pos == 0 or not text_lower[pos - 1].isalpha()
            after_ok = end >= len(text_lower) or not text_lower[end].isalpha()
            if before_ok and after_ok:
                # Skip if already covered by a multi-word match
                already = any(s <= pos and end <= e for (_, s, e) in findings)
                if not already:
                    findings.append((text[pos:end], pos, end))
            idx = pos + 1

    return findings


def detect_faces(frame_paths):
    """Detect faces in frames using OpenCV Haar Cascade (fast, no extra deps).

    Returns:
        dict: {frame_filename: [(x, y, w, h), ...]}
    """
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    detections = {}

    for path in frame_paths:
        frame = cv2.imread(path)
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        frame_key = os.path.basename(path)
        bboxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        detections[frame_key] = bboxes

    total = sum(len(v) for v in detections.values())
    print(f"Detected {total} faces across {len(frame_paths)} frames")
    return detections


def cluster_faces(frame_paths, face_detections):
    """Cluster detected faces into unique persons by position/appearance.

    Uses simple spatial clustering: faces near the same normalized position
    across frames are the same person. Returns unique face info with thumbnails.

    Returns:
        dict: {cluster_id: {center_x, center_y, count, thumbnail_b64, frames}}
    """
    all_faces = []
    for path in frame_paths:
        frame_key = os.path.basename(path)
        bboxes = face_detections.get(frame_key, [])
        frame = cv2.imread(path)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        for (fx, fy, fw, fh) in bboxes:
            cx = (fx + fw / 2) / w  # normalized center
            cy = (fy + fh / 2) / h
            all_faces.append({
                'cx': cx, 'cy': cy,
                'x': int(fx), 'y': int(fy), 'w': int(fw), 'h': int(fh),
                'frame_path': path, 'frame_key': frame_key,
                'frame_w': w, 'frame_h': h,
            })

    if not all_faces:
        return {}

    # Simple greedy clustering by normalized center distance
    clusters = {}
    cluster_id = 0
    assigned = [False] * len(all_faces)
    THRESHOLD = 0.15  # normalized distance threshold

    for i, face in enumerate(all_faces):
        if assigned[i]:
            continue
        cluster = [face]
        assigned[i] = True
        for j in range(i + 1, len(all_faces)):
            if assigned[j]:
                continue
            dist = ((face['cx'] - all_faces[j]['cx'])**2 + (face['cy'] - all_faces[j]['cy'])**2) ** 0.5
            if dist < THRESHOLD:
                cluster.append(all_faces[j])
                assigned[j] = True

        # Pick the best thumbnail (largest face)
        best = max(cluster, key=lambda f: f['w'] * f['h'])
        frame = cv2.imread(best['frame_path'])
        x, y, w, h = best['x'], best['y'], best['w'], best['h']
        # Expand crop slightly for context
        pad = int(max(w, h) * 0.2)
        y1 = max(0, y - pad)
        x1 = max(0, x - pad)
        y2 = min(best['frame_h'], y + h + pad)
        x2 = min(best['frame_w'], x + w + pad)
        crop = frame[y1:y2, x1:x2]

        if crop.size > 0:
            thumb = cv2.resize(crop, (120, 120))
            _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            thumb_b64 = base64.b64encode(buf).decode()
        else:
            thumb_b64 = ""

        avg_cx = sum(f['cx'] for f in cluster) / len(cluster)
        avg_cy = sum(f['cy'] for f in cluster) / len(cluster)

        clusters[cluster_id] = {
            'center_x': avg_cx,
            'center_y': avg_cy,
            'count': len(cluster),
            'thumbnail_b64': thumb_b64,
            'frames': list(set(f['frame_key'] for f in cluster)),
        }
        cluster_id += 1

    print(f"Clustered {len(all_faces)} face detections into {len(clusters)} unique persons")
    return clusters


def _get_face_model_path():
    """Download the face detection model if not already cached."""
    import urllib.request
    model_dir = os.path.join("/tmp", "privacylens_models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "blaze_face_short_range.tflite")
    if not os.path.exists(model_path):
        print("  Downloading face detection model...")
        url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        urllib.request.urlretrieve(url, model_path)
    return model_path


def ocr_and_detect_pii(frame_paths):
    """Run OCR on frames and detect PII with regex.

    Returns:
        dict: {frame_filename: [(x, y, w, h), ...]} for PII regions.
    """
    import easyocr

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    pii_detections = {}

    for path in frame_paths:
        frame = cv2.imread(path)
        if frame is None:
            continue

        frame_key = os.path.basename(path)
        bboxes = []

        results = reader.readtext(path)
        for bbox_coords, text, confidence in results:
            if confidence < 0.3:
                continue
            found = False
            for pii_type, pattern in PII_PATTERNS.items():
                if pattern.search(text):
                    xs = [p[0] for p in bbox_coords]
                    ys = [p[1] for p in bbox_coords]
                    x = int(min(xs))
                    y = int(min(ys))
                    w = int(max(xs) - x)
                    h = int(max(ys) - y)
                    bboxes.append((x, y, w, h))
                    print(f"  PII [{pii_type}] in {frame_key}: '{text}'")
                    found = True
                    break
            # Also check for location names in OCR text
            if not found:
                locs = find_locations_in_text(text)
                if locs:
                    xs = [p[0] for p in bbox_coords]
                    ys = [p[1] for p in bbox_coords]
                    x = int(min(xs))
                    y = int(min(ys))
                    w = int(max(xs) - x)
                    h = int(max(ys) - y)
                    bboxes.append((x, y, w, h))
                    print(f"  PII [location] in {frame_key}: '{locs[0][0]}'")

        pii_detections[frame_key] = bboxes

    total = sum(len(v) for v in pii_detections.values())
    print(f"Detected {total} PII regions across {len(frame_paths)} frames")
    return pii_detections
