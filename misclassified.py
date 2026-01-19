#!/usr/bin/env python3
import json
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "original"
FILES = ["train.json", "dev.json", "test.json"]

RE_VALID = re.compile(r"(?:^|_)valid$", re.IGNORECASE)
RE_TRAIN = re.compile(r"(?:^|_)train$", re.IGNORECASE)
RE_TEST  = re.compile(r"(?:^|_)test([123])$", re.IGNORECASE)  # test1/test2/test3


def bucket(example_id: str) -> str:
    s = example_id.strip()
    if RE_VALID.search(s):
        return "validation"
    if RE_TRAIN.search(s):
        return "train"
    m = RE_TEST.search(s)
    if m:
        return f"test{m.group(1)}"
    return "other"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def expected_bucket_for_file(filename: str) -> str:
    fn = filename.lower()
    if fn == "train.json":
        return "train"
    if fn == "dev.json":
        return "validation"
    if fn == "test.json":
        return "test"  # special: any of test1/test2/test3 is OK
    raise ValueError(f"Unexpected filename: {filename}")


def compute_not_belong_pct(filename: str, counts: Counter, total: int) -> float:
    if total == 0:
        return 0.0

    exp = expected_bucket_for_file(filename)
    if exp == "train":
        belong = counts.get("train", 0)
    elif exp == "validation":
        belong = counts.get("validation", 0)
    elif exp == "test":
        belong = counts.get("test1", 0) + counts.get("test2", 0) + counts.get("test3", 0)
    else:
        belong = 0

    not_belong = total - belong
    return 100.0 * not_belong / total


def original_split_for_file(filename: str) -> str:
    """Map a filename to the split name."""
    fn = filename.lower()
    if fn == "train.json":
        return "train"
    if fn == "dev.json":
        return "test1"
    if fn == "test.json":
        return "test2"
    raise ValueError(f"Unexpected filename: {filename}")


def generate_reallocation_map(output_path: Path = None):
    """
    Generate a JSON file mapping splits to their correct list of IDs.
    
    Logic:
    - In train.json: train suffix and primock ("other") stay in train;
      non-train samples are evenly distributed to test1 and test2
    - In dev.json/test.json: train suffix moves to train;
      all other samples stay in their original split
    """
    if output_path is None:
        output_path = Path(__file__).parent / "data" / "reallocation_map.json"
    
    # Collect all samples with their original file and bucket
    # Structure: {filename: [(id, bucket), ...]}
    samples_by_file = {}
    original_counts = {"train": 0, "test1": 0, "test2": 0}
    
    for filename in FILES:
        data = load_json(DATA_DIR / filename)
        orig_split = original_split_for_file(filename)
        original_counts[orig_split] = len(data)
        samples_by_file[filename] = [(rec["id"], bucket(rec["id"])) for rec in data]
    
    # =========================================================================
    # Build expected sets for verification BEFORE allocation
    # =========================================================================
    
    # From train.json
    train_json_train_or_other = {sid for sid, bkt in samples_by_file["train.json"] 
                                  if bkt in ("train", "other")}
    train_json_misplaced = [(sid, bkt) for sid, bkt in samples_by_file["train.json"] 
                            if bkt not in ("train", "other")]
    train_json_misplaced.sort(key=lambda x: x[0])  # Sort for deterministic distribution
    
    # From dev.json (old test1)
    dev_json_train = {sid for sid, bkt in samples_by_file["dev.json"] if bkt == "train"}
    dev_json_non_train = {sid for sid, bkt in samples_by_file["dev.json"] if bkt != "train"}
    
    # From test.json (old test2)
    test_json_train = {sid for sid, bkt in samples_by_file["test.json"] if bkt == "train"}
    test_json_non_train = {sid for sid, bkt in samples_by_file["test.json"] if bkt != "train"}
    
    # Evenly distribute misplaced from train.json
    misplaced_to_test1 = {train_json_misplaced[i][0] for i in range(0, len(train_json_misplaced), 2)}
    misplaced_to_test2 = {train_json_misplaced[i][0] for i in range(1, len(train_json_misplaced), 2)}
    
    # Expected final sets
    expected_train = train_json_train_or_other | dev_json_train | test_json_train
    expected_test1 = dev_json_non_train | misplaced_to_test1
    expected_test2 = test_json_non_train | misplaced_to_test2
    
    # =========================================================================
    # Perform allocation (same logic, but now we can verify)
    # =========================================================================
    
    all_samples = []
    for filename in FILES:
        for sid, bkt in samples_by_file[filename]:
            all_samples.append((sid, filename, bkt))
    all_samples.sort(key=lambda x: x[0])
    
    allocation = {"train": [], "test1": [], "test2": []}
    train_misplaced_list = []
    
    for sample_id, original_file, bkt in all_samples:
        orig_split = original_split_for_file(original_file)
        
        if orig_split == "train":
            if bkt == "train" or bkt == "other":
                allocation["train"].append(sample_id)
            else:
                train_misplaced_list.append(sample_id)
        else:
            if bkt == "train":
                allocation["train"].append(sample_id)
            else:
                allocation[orig_split].append(sample_id)
    
    # Evenly distribute misplaced samples from train.json to test1 and test2
    train_misplaced_list.sort()
    for i, sample_id in enumerate(train_misplaced_list):
        if i % 2 == 0:
            allocation["test1"].append(sample_id)
        else:
            allocation["test2"].append(sample_id)
    
    # Sort IDs within each split for determinism
    for split in allocation:
        allocation[split].sort()
    
    # =========================================================================
    # VERIFICATION: Check allocation matches expected
    # =========================================================================
    
    actual_train = set(allocation["train"])
    actual_test1 = set(allocation["test1"])
    actual_test2 = set(allocation["test2"])
    
    errors = []
    
    # Verify train split
    if actual_train != expected_train:
        missing = expected_train - actual_train
        extra = actual_train - expected_train
        errors.append(f"TRAIN mismatch: missing {len(missing)}, extra {len(extra)}")
        if missing:
            errors.append(f"  Missing from train: {list(missing)[:5]}...")
        if extra:
            errors.append(f"  Extra in train: {list(extra)[:5]}...")
    
    # Verify test1 split
    if actual_test1 != expected_test1:
        missing = expected_test1 - actual_test1
        extra = actual_test1 - expected_test1
        errors.append(f"TEST1 mismatch: missing {len(missing)}, extra {len(extra)}")
        if missing:
            errors.append(f"  Missing from test1: {list(missing)[:5]}...")
        if extra:
            errors.append(f"  Extra in test1: {list(extra)[:5]}...")
    
    # Verify test2 split
    if actual_test2 != expected_test2:
        missing = expected_test2 - actual_test2
        extra = actual_test2 - expected_test2
        errors.append(f"TEST2 mismatch: missing {len(missing)}, extra {len(extra)}")
        if missing:
            errors.append(f"  Missing from test2: {list(missing)[:5]}...")
        if extra:
            errors.append(f"  Extra in test2: {list(extra)[:5]}...")
    
    # Verify component counts
    print("\n--- Verification ---")
    print(f"  train.json (train+PriMock57) -> new train: {len(train_json_train_or_other)}")
    print(f"  dev.json (train) -> new train: {len(dev_json_train)}")
    print(f"  test.json (train) -> new train: {len(test_json_train)}")
    print(f"  Expected new train total: {len(expected_train)}, Actual: {len(actual_train)}")
    print()
    print(f"  dev.json (non-train) -> new test1: {len(dev_json_non_train)}")
    print(f"  train.json misplaced -> test1 (half): {len(misplaced_to_test1)}")
    print(f"  Expected new test1 total: {len(expected_test1)}, Actual: {len(actual_test1)}")
    print()
    print(f"  test.json (non-train) -> new test2: {len(test_json_non_train)}")
    print(f"  train.json misplaced -> test2 (half): {len(misplaced_to_test2)}")
    print(f"  Expected new test2 total: {len(expected_test2)}, Actual: {len(actual_test2)}")
    
    if errors:
        print("\n  ERRORS FOUND:")
        for e in errors:
            print(f"    {e}")
        raise AssertionError("Reallocation verification failed!")
    else:
        print("\n  All verifications PASSED!")
    
    # =========================================================================
    # Output
    # =========================================================================
    
    new_counts = {k: len(v) for k, v in allocation.items()}
    delta = {k: new_counts[k] - original_counts[k] for k in original_counts}
    
    counts_before = {**original_counts, "total": sum(original_counts.values())}
    counts_after = {**new_counts, "total": sum(new_counts.values())}
    
    # Add metadata
    output = {
        "note": "Primock stays in original split; non-train in train.json evenly distributed to test1/test2",
        "counts_before": counts_before,
        "counts_after": counts_after,
        "delta": delta,
        "splits": allocation,
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    # Also save the original map showing which IDs were in which original splits
    original_map = {
        "train": sorted([sid for sid, _ in samples_by_file["train.json"]]),
        "test1": sorted([sid for sid, _ in samples_by_file["dev.json"]]),
        "test2": sorted([sid for sid, _ in samples_by_file["test.json"]]),
    }
    original_map_path = output_path.parent / "original_map.json"
    original_output = {
        "note": "Original split assignments before reallocation",
        "counts": {k: len(v) for k, v in original_map.items()},
        "splits": original_map,
    }
    with open(original_map_path, "w") as f:
        json.dump(original_output, f, indent=2)
    
    print(f"\nOriginal map saved to: {original_map_path}")
    print(f"Reallocation map saved to: {output_path}")
    print(f"\n  Split       Before  After   Change")
    print(f"  ----------  ------  -----   ------")
    split_display = [("train", "train"), ("dev/test1", "test1"), ("test/test2", "test2")]
    for display_name, split in split_display:
        before = original_counts[split]
        after = new_counts[split]
        diff = after - before
        sign = "+" if diff >= 0 else ""
        print(f"  {display_name:<10} {before:>6}  {after:>5}   {sign}{diff}")
    print(f"  ----------  ------  -----   ------")
    print(f"  {'total':<10} {sum(original_counts.values()):>6}  {sum(new_counts.values()):>5}")
    
    # Generate reallocation breakdown table (similar to original breakdown)
    # Build a map from sample_id to bucket for quick lookup
    all_buckets = {}
    for filename in FILES:
        for sid, bkt in samples_by_file[filename]:
            all_buckets[sid] = bkt
    
    print("\n--- Reallocation Breakdown ---")
    print("New Split   Total  Train  Valid  Test1  Test2  Test3  PriMock57")
    print("----------  -----  -----  -----  -----  -----  -----  ---------")
    
    realloc_breakdown = {}
    for split_name in ["train", "test1", "test2"]:
        split_ids = allocation[split_name]
        counts = Counter(all_buckets[sid] for sid in split_ids)
        row = {
            "total": len(split_ids),
            "train": counts.get("train", 0),
            "validation": counts.get("validation", 0),
            "test1": counts.get("test1", 0),
            "test2": counts.get("test2", 0),
            "test3": counts.get("test3", 0),
            "other": counts.get("other", 0),
        }
        realloc_breakdown[split_name] = row
        print(f"{split_name:<10}  {row['total']:>5}  {row['train']:>5}  {row['validation']:>5}  {row['test1']:>5}  {row['test2']:>5}  {row['test3']:>5}  {row['other']:>9}")
    
    # Save reallocation breakdown to JSON
    realloc_breakdown_path = output_path.parent / "reallocation_breakdown.json"
    with open(realloc_breakdown_path, "w") as f:
        json.dump(realloc_breakdown, f, indent=2)
    print(f"\nReallocation breakdown saved to: {realloc_breakdown_path}")
    
    return allocation


def main():
    rows = []
    for filename in FILES:
        data = load_json(DATA_DIR / filename)
        ids = [rec["id"] for rec in data]

        counts = Counter(bucket(i) for i in ids)

        rows.append({
            "file": filename,
            "total": len(ids),
            "train": counts.get("train", 0),
            "validation": counts.get("validation", 0),
            "test1": counts.get("test1", 0),
            "test2": counts.get("test2", 0),
            "test3": counts.get("test3", 0),
            "other": counts.get("other", 0),
        })

    headers = ["File", "Total", "Train", "Valid", "Test1", "Test2", "Test3", "PriMock57"]
    keys    = ["file", "total", "train", "validation", "test1", "test2", "test3", "other"]

    widths = []
    for h, k in zip(headers, keys):
        w = len(h)
        for r in rows:
            w = max(w, len(str(r[k])))
        widths.append(w)

    def fmt_row(values):
        out = []
        for v, w in zip(values, widths):
            if isinstance(v, int):
                out.append(str(v).rjust(w))
            else:
                out.append(str(v).ljust(w))
        return "  ".join(out)

    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt_row([r[k] for k in keys]))

    # Generate reallocation map
    generate_reallocation_map()


if __name__ == "__main__":
    main()
