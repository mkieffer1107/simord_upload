#!/usr/bin/env python3
"""
Upload SIMORD dataset to HuggingFace Hub.

This script:
1. Loads the reallocation map to determine which IDs go into each split
2. Loads transcripts from ACI-Bench dataset (virtassist, aci, virtscribe subsets)
3. Loads expected_orders from the original SIMORD data files
4. Creates a new dataset with: id, transcript, orders
5. Saves to data/new/ and optionally pushes to HuggingFace
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset, Dataset, DatasetDict, Features, Value, Sequence
from huggingface_hub import create_repo

from textgrid_to_transcript import load_all_primock_transcripts, parse_primock_id

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

def str_to_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}

parser = argparse.ArgumentParser(description="Format SIMORD data and push to HF")
parser.add_argument("--hf_username", default=os.environ.get("HF_USERNAME", "mkieffer"))
parser.add_argument("--hf_repo_name", default=os.environ.get("HF_REPO_NAME", "SIMORD"))
parser.add_argument("--private", default=os.environ.get("PRIVATE", "false"),
                    help="Whether the HF dataset repo should be private (true/false)")
args = parser.parse_args()

HF_USERNAME = args.hf_username
HF_REPO_NAME = args.hf_repo_name
PRIVATE = str_to_bool(args.private)
HF_REPO_ID = f"{HF_USERNAME}/{HF_REPO_NAME}"

DATA_DIR = Path(__file__).parent / "data"
ORIGINAL_DIR = DATA_DIR / "original"
NEW_DIR = DATA_DIR / "new"
NEW_DIR.mkdir(parents=True, exist_ok=True)

# ACI-Bench subsets and splits
ACI_SUBSETS = ["virtassist", "aci", "virtscribe"]
ACI_SPLITS = ["train", "valid", "test1", "test2", "test3"]  # Actual split names in HF dataset

# ---------------------------------------------------------------------------
# Parsing SIMORD IDs
# ---------------------------------------------------------------------------

# Pattern: acibench_D2N001_virtassist_train or acibench_D2N088_virtassist_clinicalnlp_taskB_test1
# We need to extract: encounter_id, subset
# Format is: acibench_{encounter_id}_{subset}_{...rest including split info}
ACIBENCH_PATTERN = re.compile(
    r"^acibench_(D2N\d+)_(virtassist|aci|virtscribe)_(.+)$",
    re.IGNORECASE
)
PRIMOCK_PATTERN = re.compile(r"^primock57_", re.IGNORECASE)


def parse_acibench_id(simord_id: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse an acibench SIMORD ID.
    
    Returns (encounter_id, subset, split_info) or None if not acibench.
    
    Examples:
      "acibench_D2N001_virtassist_train" -> ("D2N001", "virtassist", "train")
      "acibench_D2N088_virtassist_clinicalnlp_taskB_test1" -> ("D2N088", "virtassist", "clinicalnlp_taskB_test1")
    """
    m = ACIBENCH_PATTERN.match(simord_id)
    if m:
        return m.group(1), m.group(2).lower(), m.group(3).lower()
    return None


# Pattern to extract split from the end of split_info (train, valid, test1, test2, test3)
SPLIT_SUFFIX_PATTERN = re.compile(r"(train|valid|test[123])$", re.IGNORECASE)


def extract_split_from_info(split_info: str) -> str:
    """
    Extract the split name from split_info.
    
    Examples:
      "train" -> "train"
      "clinicalnlp_taskB_test1" -> "test1"
      "clef_taskC_test3" -> "test3"
    """
    m = SPLIT_SUFFIX_PATTERN.search(split_info)
    if m:
        return m.group(1).lower()
    return split_info  # fallback to full split_info


def is_primock(simord_id: str) -> bool:
    """Check if this is a primock ID."""
    return bool(PRIMOCK_PATTERN.match(simord_id))


def get_all_primock_ids(reallocation: Dict) -> List[str]:
    """Extract all primock IDs from the reallocation map."""
    primock_ids = []
    for split_name, ids in reallocation.get("splits", {}).items():
        for simord_id in ids:
            if is_primock(simord_id):
                primock_ids.append(simord_id)
    return primock_ids


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_reallocation_map() -> Dict:
    """Load the reallocation map from data/reallocation_map.json."""
    path = DATA_DIR / "reallocation_map.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_original_orders() -> Tuple[Dict[str, List[Dict]], Dict[str, int]]:
    """
    Load all original SIMORD data and build a map from ID to expected_orders.
    
    Returns: ({simord_id: expected_orders_list}, {split_name: count})
    """
    orders_map = {}
    original_counts = {}
    
    display_names = {
        "train": "train",
        "dev": "dev/test1",
        "test": "test/test2"
    }
    
    print(f"  Original dataset distribution:")
    print(f"  {'-'*25}")
    print(f"  {'Split':<15} {'Count':>10}")
    print(f"  {'-'*25}")
    
    total = 0
    for filename in ["train.json", "dev.json", "test.json"]:
        filepath = ORIGINAL_DIR / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping")
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        count = len(data)
        split_key = filename.split('.')[0]
        original_counts[split_key] = count
        display_name = display_names.get(split_key, split_key)

        print(f"  {display_name:<15} {count:>10}")
        total += count
        
        for entry in data:
            orders_map[entry["id"]] = entry.get("expected_orders", [])
            
    print(f"  {'-'*25}")
    print(f"  {'TOTAL':<15} {total:>10}")
            
    return orders_map, original_counts


def load_acibench_transcripts() -> Dict[Tuple[str, str], str]:
    """
    Load all ACI-Bench transcripts from HuggingFace.
    
    Returns: {(encounter_id, subset): dialogue}
    
    We key by (encounter_id, subset) since encounter_id is unique within a subset.
    """
    transcripts = {}
    
    print("Loading ACI-Bench dataset from HuggingFace...")
    for subset in ACI_SUBSETS:
        print(f"  Loading subset: {subset}")
        try:
            ds = load_dataset("mkieffer/ACI-Bench", subset)
        except Exception as e:
            print(f"    Error loading {subset}: {e}")
            continue
        
        for split in ACI_SPLITS:
            if split not in ds:
                print(f"    Split {split} not found in {subset}")
                continue
            
            split_data = ds[split]
            for row in split_data:
                encounter_id = row.get("encounter_id", "")
                dialogue = row.get("dialogue", "")
                
                # Key: (encounter_id, subset) - encounter_id is unique within subset
                key = (encounter_id.upper(), subset.lower())
                if key not in transcripts:
                    transcripts[key] = dialogue
    
    print(f"  Loaded {len(transcripts)} transcripts total")
    return transcripts


# ---------------------------------------------------------------------------
# Build new dataset
# ---------------------------------------------------------------------------

def build_simord_dataset(
    reallocation: Dict,
    orders_map: Dict[str, List[Dict]],
    acibench_transcripts: Dict[Tuple[str, str], str],
    primock_transcripts: Dict[str, str]
) -> Dict[str, List[Dict]]:
    """
    Build the new SIMORD dataset splits.
    
    Returns: {split_name: [rows]}
    """
    splits = reallocation.get("splits", {})
    result = {"train": [], "test1": [], "test2": []}
    
    stats = {
        "total": 0,
        "acibench": 0,
        "primock": 0,
        "missing_transcript": 0,
        "missing_orders": 0,
        "success": 0,
    }
    
    for split_name, ids in splits.items():
        print(f"\nProcessing split: {split_name} ({len(ids)} IDs)")
        
        for simord_id in ids:
            stats["total"] += 1
            
            # Handle primock IDs
            if is_primock(simord_id):
                stats["primock"] += 1
                
                # Get transcript from primock lookup
                transcript = primock_transcripts.get(simord_id)
                if not transcript:
                    stats["missing_transcript"] += 1
                    print(f"  Warning: No transcript for {simord_id}")
                    continue

                # Parse primock ID for the row ID (keep as primock57_X_Y)
                parsed = parse_primock_id(simord_id)
                if parsed:
                    day, consultation = parsed
                    row_id = f"primock57_{day}_{consultation}"
                else:
                    row_id = simord_id
                
            else:
                # Handle acibench IDs
                stats["acibench"] += 1
                
                parsed = parse_acibench_id(simord_id)
                if not parsed:
                    print(f"  Warning: Could not parse ID: {simord_id}")
                    continue
                
                encounter_id, subset, split_info = parsed
                
                # Get transcript using (encounter_id, subset) key
                lookup_key = (encounter_id.upper(), subset.lower())
                transcript = acibench_transcripts.get(lookup_key)
                
                if not transcript:
                    stats["missing_transcript"] += 1
                    print(f"  Warning: No transcript for {simord_id} (key: {lookup_key})")
                    continue
                
                # Extract split from split_info and build row_id
                original_split = extract_split_from_info(split_info)
                row_id = f"acibench_{subset}_{original_split}_{encounter_id}"
            
            # Get orders from original data
            orders = orders_map.get(simord_id)
            if orders is None:
                # Try case variations
                for key in orders_map:
                    if key.lower() == simord_id.lower():
                        orders = orders_map[key]
                    break

            if orders is None:
                stats["missing_orders"] += 1
                print(f"  Warning: No orders for {simord_id}")
                orders = []
            
            # Build the row
            row = {
                "id": row_id,
                "transcript": transcript,
                "orders": orders,
            }
            result[split_name].append(row)
            stats["success"] += 1
    
    print(f"\n--- Stats ---")
    print(f"  Total IDs: {stats['total']}")
    print(f"  ACI-Bench: {stats['acibench']}")
    print(f"  Primock: {stats['primock']}")
    print(f"  Missing transcript: {stats['missing_transcript']}")
    print(f"  Missing orders: {stats['missing_orders']}")
    print(f"  Success: {stats['success']}")
    
    return result


# ---------------------------------------------------------------------------
# Save and upload
# ---------------------------------------------------------------------------

def validate_counts(reallocation: Dict, data: Dict[str, List[Dict]]) -> bool:
    """
    Validate that the built dataset matches expected counts from reallocation map.
    
    Returns True if all counts match, False otherwise.
    """
    print("\n--- Validation ---")
    
    expected_splits = reallocation.get("splits", {})
    expected_total = sum(len(ids) for ids in expected_splits.values())
    actual_total = sum(len(rows) for rows in data.values())
    
    all_match = True
    
    print(f"\n  {'Split':<10} {'Expected':>10} {'Actual':>10} {'Status':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for split_name in ["train", "test1", "test2"]:
        expected = len(expected_splits.get(split_name, []))
        actual = len(data.get(split_name, []))
        status = "✓" if expected == actual else "✗ MISMATCH"
        if expected != actual:
            all_match = False
        print(f"  {split_name:<10} {expected:>10} {actual:>10} {status:>10}")
    
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'TOTAL':<10} {expected_total:>10} {actual_total:>10}", end="")
    
    if expected_total == actual_total:
        print(f" {'✓':>10}")
        print(f"\n  All {actual_total} entries accounted for!")
    else:
        print(f" {'✗ MISMATCH':>10}")
        print(f"\n  WARNING: Missing {expected_total - actual_total} entries!")
        all_match = False
    
    return all_match


def save_to_disk(data: Dict[str, List[Dict]]):
    """Save the dataset splits to data/new/."""
    for split_name, rows in data.items():
        filepath = NEW_DIR / f"{split_name}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(rows)} rows to {filepath}")


def push_to_hub(data: Dict[str, List[Dict]]):
    """Push the dataset to HuggingFace Hub."""
    print(f"\nPushing dataset to HuggingFace Hub as {HF_REPO_ID} (private={PRIVATE})...")

    create_repo(HF_REPO_ID, repo_type="dataset", private=PRIVATE, exist_ok=True)

    # Convert to HuggingFace datasets
    dsd = DatasetDict()
    for split_name, rows in data.items():
        if rows:
            dsd[split_name] = Dataset.from_list(rows)

    dsd.push_to_hub(HF_REPO_ID, private=PRIVATE)
    print(f"Dataset pushed to https://huggingface.co/datasets/{HF_REPO_ID}")


def print_delta_table(original_counts: Dict[str, int], new_data: Dict[str, List[Dict]]):
    """
    Print a table showing the changes in dataset sizes.
    """
    print("\n--- Changes Summary ---")
    print(f"  {'Split':<15} {'Original':>10} {'New':>10} {'Delta':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
    
    comparisons = [
        ("train", "train"),
        ("dev", "test1"),
        ("test", "test2")
    ]
    
    total_orig = 0
    total_new = 0
    
    for orig_key, new_key in comparisons:
        orig_count = original_counts.get(orig_key, 0)
        new_count = len(new_data.get(new_key, []))
        
        delta = new_count - orig_count
        delta_str = f"{delta:+d}" if delta != 0 else "-"
        
        display_label = f"{orig_key}/{new_key}"
        if orig_key == "train": display_label = "train"
        elif orig_key == "dev": display_label = "dev/test1"
        elif orig_key == "test": display_label = "test/test2"

        print(f"  {display_label:<15} {orig_count:>10} {new_count:>10} {delta_str:>10}")
        
        total_orig += orig_count
        total_new += new_count
        
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
    
    total_delta = total_new - total_orig
    total_delta_str = f"{total_delta:+d}" if total_delta != 0 else "-"
    print(f"  {'TOTAL':<15} {total_orig:>10} {total_new:>10} {total_delta_str:>10}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("SIMORD Dataset Builder")
    print("=" * 60)
    
    # Load all required data
    print("\n1. Loading reallocation map...")
    reallocation = load_reallocation_map()
    print(f"   Splits: {list(reallocation.get('splits', {}).keys())}")
    
    print("\n2. Loading original SIMORD orders...")
    orders_map, original_counts = load_original_orders()
    print(f"   Loaded orders for {len(orders_map)} entries")
    
    print("\n3. Loading ACI-Bench transcripts...")
    acibench_transcripts = load_acibench_transcripts()
    
    print("\n4. Loading Primock57 transcripts from GitHub...")
    primock_ids = get_all_primock_ids(reallocation)
    print(f"   Found {len(primock_ids)} primock IDs")
    primock_transcripts = load_all_primock_transcripts(primock_ids)
    
    print("\n5. Building SIMORD dataset...")
    data = build_simord_dataset(reallocation, orders_map, acibench_transcripts, primock_transcripts)
    
    print("\n6. Validating counts...")
    validate_counts(reallocation, data)
    
    print("\n7. Summary of Changes...")
    print_delta_table(original_counts, data)
    
    print("\n8. Saving to disk...")
    save_to_disk(data)
    
    print("\n8. Pushing to HuggingFace Hub...")
    # push_to_hub(data)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
