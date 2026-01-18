#!/usr/bin/env python3
"""
Convert Praat TextGrid files to transcript format.

Based on: https://github.com/babylonhealth/primock57/blob/main/scripts/textgrid_to_transcript.py

TextGrid files contain timed utterances from doctor and patient recordings.
This module combines them into a single transcript with speaker tags.
"""

import re
from typing import List, Dict, Optional
from urllib.request import urlopen
from urllib.error import URLError, HTTPError


# ---------------------------------------------------------------------------
# TextGrid parsing
# ---------------------------------------------------------------------------

def parse_textgrid(content: str) -> List[Dict]:
    """
    Parse a TextGrid file content and extract utterances.
    
    Returns a list of dicts with keys: 'from', 'to', 'text'
    """
    utterances = []
    
    # Find all interval blocks
    # Format:
    # intervals [N]:
    #     xmin = 0.359362449876447
    #     xmax = 4.80802814026118
    #     text = "Hello."
    
    interval_pattern = re.compile(
        r'intervals\s*\[\d+\]:\s*'
        r'xmin\s*=\s*([\d.]+)\s*'
        r'xmax\s*=\s*([\d.]+)\s*'
        r'text\s*=\s*"([^"]*)"',
        re.MULTILINE | re.DOTALL
    )
    
    for match in interval_pattern.finditer(content):
        xmin = float(match.group(1))
        xmax = float(match.group(2))
        text = match.group(3).strip()
        
        # Skip empty utterances
        if text:
            utterances.append({
                'from': xmin,
                'to': xmax,
                'text': text
            })
    
    return utterances


def strip_transcript_tags(text: str) -> str:
    """
    Remove annotation tags from transcript text.
    
    Tags like <UNSURE>word</UNSURE>, <INAUDIBLE_SPEECH/>, <UNIN/>, etc.
    """
    if not text:
        return ""
    
    # Remove self-closing tags like <INAUDIBLE_SPEECH/>, <UNIN/>
    text = re.sub(r'<[^>]+/>', '', text)
    
    # Remove paired tags but keep content: <UNSURE>word</UNSURE> -> word
    text = re.sub(r'<([^>/]+)>([^<]*)</\1>', r'\2', text)
    
    # Clean up any remaining tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def get_utterances_textgrid(content: str) -> List[Dict]:
    """
    Parse TextGrid content and return cleaned utterances.
    """
    utterances = parse_textgrid(content)
    
    # Clean up text
    for u in utterances:
        u['text'] = strip_transcript_tags(u['text'])
    
    # Filter out empty utterances after cleaning
    utterances = [u for u in utterances if u['text']]
    
    return utterances


# ---------------------------------------------------------------------------
# Transcript combination
# ---------------------------------------------------------------------------

def get_combined_transcript(doctor_content: str, patient_content: str) -> str:
    """
    Combine doctor and patient TextGrid contents into a single transcript.
    
    Returns a transcript string in the same format as ACI-Bench:
    [doctor] Hello...
    [patient] Hi...
    """
    utterances_doctor = get_utterances_textgrid(doctor_content)
    utterances_patient = get_utterances_textgrid(patient_content)
    
    for u in utterances_doctor:
        u['speaker'] = 'doctor'
    for u in utterances_patient:
        u['speaker'] = 'patient'
    
    combined_utterances = utterances_doctor + utterances_patient
    combined_utterances.sort(key=lambda x: x['from'])
    
    # Format like ACI-Bench: [speaker] text
    lines = [f"[{u['speaker']}] {u['text']}" for u in combined_utterances]
    
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# GitHub fetching
# ---------------------------------------------------------------------------

PRIMOCK_BASE_URL = "https://raw.githubusercontent.com/babylonhealth/primock57/main/transcripts"


def fetch_textgrid(url: str) -> Optional[str]:
    """
    Fetch a TextGrid file from a URL.
    
    Returns the content as a string, or None if fetch fails.
    """
    try:
        with urlopen(url, timeout=30) as response:
            return response.read().decode('utf-8')
    except (URLError, HTTPError) as e:
        print(f"  Error fetching {url}: {e}")
        return None


def parse_primock_id(simord_id: str) -> Optional[tuple]:
    """
    Parse a primock SIMORD ID.
    
    Format: primock57_X_Y where X is day number, Y is consultation number
    Example: primock57_4_1 -> (4, 1)
    
    Returns (day, consultation) or None if not a primock ID.
    """
    match = re.match(r'^primock57_(\d+)_(\d+)$', simord_id, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def get_primock_transcript(simord_id: str) -> Optional[str]:
    """
    Fetch and combine primock57 transcript from GitHub.
    
    Args:
        simord_id: ID like "primock57_4_1"
        
    Returns:
        Combined transcript string, or None if fetch fails.
    """
    parsed = parse_primock_id(simord_id)
    if not parsed:
        return None
    
    day, consultation = parsed
    
    # Build filenames: day4_consultation01_doctor.TextGrid
    filename_base = f"day{day}_consultation{consultation:02d}"
    
    doctor_url = f"{PRIMOCK_BASE_URL}/{filename_base}_doctor.TextGrid"
    patient_url = f"{PRIMOCK_BASE_URL}/{filename_base}_patient.TextGrid"
    
    # Fetch both files
    doctor_content = fetch_textgrid(doctor_url)
    patient_content = fetch_textgrid(patient_url)
    
    if doctor_content is None or patient_content is None:
        return None
    
    return get_combined_transcript(doctor_content, patient_content)


def load_all_primock_transcripts(primock_ids: List[str]) -> Dict[str, str]:
    """
    Load all primock transcripts for a list of IDs.
    
    Returns: {simord_id: transcript}
    """
    transcripts = {}
    
    print(f"Loading {len(primock_ids)} primock transcripts from GitHub...")
    
    for i, simord_id in enumerate(primock_ids):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(primock_ids)}")
        
        transcript = get_primock_transcript(simord_id)
        if transcript:
            transcripts[simord_id] = transcript
        else:
            print(f"  Warning: Failed to load transcript for {simord_id}")
    
    print(f"  Loaded {len(transcripts)} primock transcripts")
    return transcripts


# ---------------------------------------------------------------------------
# CLI for testing
# ---------------------------------------------------------------------------

def main():
    """Test the transcript conversion."""
    import sys
    
    if len(sys.argv) > 1:
        simord_id = sys.argv[1]
    else:
        simord_id = "primock57_4_1"
    
    print(f"Fetching transcript for: {simord_id}")
    transcript = get_primock_transcript(simord_id)
    
    if transcript:
        print("\n" + "=" * 60)
        print(transcript[:2000])
        if len(transcript) > 2000:
            print(f"\n... ({len(transcript)} chars total)")
    else:
        print("Failed to fetch transcript")


if __name__ == "__main__":
    main()

