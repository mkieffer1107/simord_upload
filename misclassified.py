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


def correct_split_for_bucket(bkt: str) -> str:
    """Map a detected bucket to the correct split name."""
    if bkt == "train":
        return "train"
    if bkt == "validation":
        return "test1"
    if bkt in ("test1", "test2", "test3"):
        return "test2"
    return "other"  # will be handled specially


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
    if output_path is None:
        output_path = Path(__file__).parent / "data" / "reallocation_map.json"
    """
    Generate a JSON file mapping splits to their correct list of IDs.
    
    Logic:
    - IDs with _train suffix -> train split
    - IDs with _valid suffix -> dev split
    - IDs with _test1/_test2/_test3 suffix -> test split
    - IDs in "other" bucket -> stay in their original file's split
    """
    # Collect all samples with their original file
    all_samples = []  # list of (id, original_file)
    original_counts = {"train": 0, "test1": 0, "test2": 0}
    
    for filename in FILES:
        data = load_json(DATA_DIR / filename)
        orig_split = original_split_for_file(filename)
        original_counts[orig_split] = len(data)
        for rec in data:
            all_samples.append((rec["id"], filename))
    
    # Sort deterministically by ID
    all_samples.sort(key=lambda x: x[0])
    
    # Allocate each sample to the correct split
    allocation = {"train": [], "test1": [], "test2": []}
    
    for sample_id, original_file in all_samples:
        bkt = bucket(sample_id)
        
        if bkt == "other":
            # Keep in original split
            correct_split = original_split_for_file(original_file)
        else:
            # Move to correct split based on ID pattern
            correct_split = correct_split_for_bucket(bkt)
        
        allocation[correct_split].append(sample_id)
    
    # Sort IDs within each split for determinism
    for split in allocation:
        allocation[split].sort()
    
    new_counts = {k: len(v) for k, v in allocation.items()}
    delta = {k: new_counts[k] - original_counts[k] for k in original_counts}
    
    counts_before = {**original_counts, "total": sum(original_counts.values())}
    counts_after = {**new_counts, "total": sum(new_counts.values())}
    
    # Add metadata
    output = {
        "note": "Samples in 'other' bucket (primock57 dataset) remain in their original split",
        "counts_before": counts_before,
        "counts_after": counts_after,
        "delta": delta,
        "splits": allocation,
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nReallocation map saved to: {output_path}")
    print(f"\n  Split       Before  After   Change")
    print(f"  ----------  ------  -----   ------")
    # Show mapping from original file name to new split name
    split_display = [("train", "train"), ("dev/test1", "test1"), ("test/test2", "test2")]
    for display_name, split in split_display:
        before = original_counts[split]
        after = new_counts[split]
        diff = after - before
        sign = "+" if diff >= 0 else ""
        print(f"  {display_name:<10} {before:>6}  {after:>5}   {sign}{diff}")
    print(f"  ----------  ------  -----   ------")
    print(f"  {'total':<10} {sum(original_counts.values()):>6}  {sum(new_counts.values()):>5}")
    
    return allocation


def main():
    rows = []
    for filename in FILES:
        data = load_json(DATA_DIR / filename)
        ids = [rec["id"] for rec in data]

        counts = Counter(bucket(i) for i in ids)
        pct_not_belong = compute_not_belong_pct(filename, counts, len(ids))

        rows.append({
            "file": filename,
            "total": len(ids),
            "train": counts.get("train", 0),
            "validation": counts.get("validation", 0),
            "test1": counts.get("test1", 0),
            "test2": counts.get("test2", 0),
            "test3": counts.get("test3", 0),
            "other": counts.get("other", 0),
            "pct_not_belong": f"{pct_not_belong:.1f}%",
        })

    headers = ["File", "Total", "Train", "Valid", "Test1", "Test2", "Test3", "Other", "%NotBelong"]
    keys    = ["file", "total", "train", "validation", "test1", "test2", "test3", "other", "pct_not_belong"]

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
