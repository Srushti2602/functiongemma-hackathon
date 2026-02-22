"""PrivacyLens — FunctionGemma-powered PII analysis on transcribed text.

Uses FunctionGemma (via generate_hybrid) to:
1. Plan the analysis pipeline from user instruction
2. Classify detected PII types and risk levels
3. Generate a structured privacy report
"""

import sys
import os
sys.path.insert(0, "cactus/python/src")
os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

import re
from main import generate_hybrid
from tools import AUDIO_PII_TOOLS
from detection import (
    PII_PATTERNS, find_locations_in_text,
    NAME_CONTEXT_PATTERNS, ADDRESS_CONTEXT_PATTERNS,
    SPOKEN_PHONE_RE, _COMMON_WORDS, ALL_LOCATIONS,
)
from transcribe import format_timestamp


def plan_analysis(instruction):
    """Ask FunctionGemma what tools to use for this instruction.

    Returns the planned tool calls so the UI can show
    'FunctionGemma decided to: transcribe → detect PII → classify risk → report'
    """
    messages = [{"role": "user", "content": instruction}]
    result = generate_hybrid(messages, AUDIO_PII_TOOLS)
    planned = [c["name"] for c in result.get("function_calls", [])]
    source = result.get("source", "unknown")
    return {
        "planned_tools": planned,
        "source": source,
        "raw_calls": result.get("function_calls", []),
        "time_ms": result.get("total_time_ms", 0),
    }


def detect_pii_in_segments(segments):
    """Run regex PII detection on each transcript segment.

    Returns:
        list[dict]: [{start, end, text, pii_type, matched_text}, ...]
    """
    findings = []

    for seg in segments:
        text = seg["text"]
        for pii_type, pattern in PII_PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "timestamp": format_timestamp(seg["start"]),
                    "pii_type": pii_type,
                    "matched_text": match.group(),
                    "context": text,
                })

    # --- Person name detection (contextual patterns only) ---
    for seg in segments:
        text = seg["text"]
        for pat in NAME_CONTEXT_PATTERNS:
            for match in pat.finditer(text):
                name = match.group(1).strip()
                words = name.split()
                # Filter out common/stop words that leak through IGNORECASE
                words = [w for w in words if w.lower() not in _COMMON_WORDS
                         and w.lower() not in ALL_LOCATIONS and len(w) > 1]
                if words:
                    findings.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "timestamp": format_timestamp(seg["start"]),
                        "pii_type": "person_name",
                        "matched_text": " ".join(words),
                        "context": text,
                    })

    # --- Spoken phone number detection ("five five five one two three...") ---
    for seg in segments:
        for match in SPOKEN_PHONE_RE.finditer(seg["text"]):
            findings.append({
                "start": seg["start"],
                "end": seg["end"],
                "timestamp": format_timestamp(seg["start"]),
                "pii_type": "phone",
                "matched_text": match.group().strip(),
                "context": seg["text"],
            })

    # --- Address detection from speech context ---
    for seg in segments:
        for pat in ADDRESS_CONTEXT_PATTERNS:
            for match in pat.finditer(seg["text"]):
                addr = match.group(1).strip()
                if len(addr) > 3:
                    findings.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "timestamp": format_timestamp(seg["start"]),
                        "pii_type": "address",
                        "matched_text": addr,
                        "context": seg["text"],
                    })

    # --- Location name detection (countries, cities, states) ---
    for seg in segments:
        locs = find_locations_in_text(seg["text"])
        for (loc_text, _, _) in locs:
            findings.append({
                "start": seg["start"],
                "end": seg["end"],
                "timestamp": format_timestamp(seg["start"]),
                "pii_type": "location",
                "matched_text": loc_text,
                "context": seg["text"],
            })

    # Deduplicate by (pii_type, matched_text)
    seen = set()
    unique = []
    for f in findings:
        key = (f["pii_type"], f["matched_text"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(f)

    return unique


def classify_risk(findings):
    """Use FunctionGemma to classify the privacy risk level.

    Falls back to heuristic if model doesn't produce a useful result.
    """
    if not findings:
        return {"level": "none", "reason": "No PII detected in audio"}

    pii_types = list(set(f["pii_type"] for f in findings))
    pii_count = len(findings)

    # Ask FunctionGemma to classify
    instruction = (
        f"Classify the privacy risk: found {pii_count} PII items "
        f"including {', '.join(pii_types)}"
    )
    messages = [{"role": "user", "content": instruction}]
    result = generate_hybrid(messages, [
        {
            "name": "classify_risk",
            "description": "Classify privacy risk level as low, medium, or high",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {"type": "string", "description": "Risk level: low, medium, or high"},
                    "reason": {"type": "string", "description": "Brief reason for the risk level"},
                },
                "required": ["level"],
            },
        }
    ])

    calls = result.get("function_calls", [])
    source = result.get("source", "unknown")

    if calls and calls[0].get("arguments", {}).get("level"):
        level = calls[0]["arguments"]["level"].lower().strip()
        reason = calls[0]["arguments"].get("reason", "")
        if level in ("low", "medium", "high"):
            return {"level": level, "reason": reason, "source": source}

    # Heuristic fallback
    high_types = {"ssn", "credit_card"}
    medium_types = {"phone", "email", "address", "person_name", "location"}

    if any(f["pii_type"] in high_types for f in findings):
        level = "high"
        reason = "Sensitive PII (SSN/credit card) detected in audio"
    elif pii_count >= 3 or any(f["pii_type"] in medium_types for f in findings):
        level = "medium"
        reason = f"{pii_count} PII items including {', '.join(pii_types)}"
    else:
        level = "low"
        reason = f"Minor PII detected: {', '.join(pii_types)}"

    return {"level": level, "reason": reason, "source": "heuristic"}


def mask_pii(text, pii_type):
    """Mask a PII value for display (show partial, hide rest)."""
    if not text:
        return "***"
    if pii_type == "phone":
        # Show last 4 digits
        digits = re.sub(r'\D', '', text)
        return "***-***-" + digits[-4:] if len(digits) >= 4 else "***"
    if pii_type == "email":
        parts = text.split("@")
        if len(parts) == 2:
            return parts[0][0] + "***@" + parts[1]
        return "***"
    if pii_type == "ssn":
        return "***-**-" + text[-4:] if len(text) >= 4 else "***"
    if pii_type == "credit_card":
        digits = re.sub(r'\D', '', text)
        return "****-****-****-" + digits[-4:] if len(digits) >= 4 else "***"
    if pii_type == "person_name":
        return text[0] + "***" if text else "***"
    if pii_type == "address":
        words = text.split()
        return words[0] + " ***" if words else "***"
    if pii_type == "license_plate":
        return text[:2] + "***" if len(text) >= 2 else "***"
    if pii_type == "location":
        return text[0] + "***" if text else "***"
    if pii_type == "ip_address":
        parts = text.split(".")
        return parts[0] + ".***.***.***" if len(parts) >= 1 else "***"
    if pii_type == "url":
        return text[:15] + "***" if len(text) > 15 else text[0] + "***"
    if pii_type == "date_of_birth":
        return "**/**/****"
    return text[0] + "***" if text else "***"


def run_full_analysis(segments, instruction="Scan for any personal information"):
    """Run the complete FunctionGemma-powered PII analysis pipeline.

    Returns:
        dict with plan, findings, risk, and report data.
    """
    # Step 1: FunctionGemma plans the analysis
    plan = plan_analysis(instruction)

    # Step 2: Detect PII in transcript segments
    findings = detect_pii_in_segments(segments)

    # Step 3: FunctionGemma classifies risk
    risk = classify_risk(findings)

    # Step 4: Build the report
    masked_findings = []
    for f in findings:
        masked_findings.append({
            **f,
            "masked": mask_pii(f["matched_text"], f["pii_type"]),
        })

    return {
        "plan": plan,
        "findings": masked_findings,
        "risk": risk,
        "total_pii": len(findings),
        "pii_types": list(set(f["pii_type"] for f in findings)),
        "transcript_segments": len(segments),
    }
