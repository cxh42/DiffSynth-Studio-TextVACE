"""
Pick a fixed-seed random subset of 30 samples from the inference metadata
for overfitting comparison. Writes:
  data/inference_new/inference_data/metadata_overfitting_subset.csv
  outputs/inference_overfitting/subset_ids.json
"""
import csv, json, random, os
from pathlib import Path

SOURCE_CSV = "data/inference_new/inference_data/metadata.csv"
SUBSET_CSV = "data/inference_new/inference_data/metadata_overfitting_subset.csv"
SUBSET_IDS_JSON = "outputs/inference_overfitting/subset_ids.json"
SUBSET_SIZE = 30
SEED = 42


def main():
    rows = list(csv.DictReader(open(SOURCE_CSV)))
    random.seed(SEED)
    sampled = random.sample(rows, SUBSET_SIZE)
    sampled.sort(key=lambda r: r["id"])  # stable order

    fieldnames = list(rows[0].keys())
    Path(SUBSET_CSV).parent.mkdir(parents=True, exist_ok=True)
    with open(SUBSET_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sampled)

    Path(SUBSET_IDS_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(SUBSET_IDS_JSON, "w") as f:
        json.dump({
            "seed": SEED,
            "size": SUBSET_SIZE,
            "ids": [r["id"] for r in sampled],
            "source_csv": SOURCE_CSV,
        }, f, indent=2)

    print(f"wrote {SUBSET_CSV} ({len(sampled)} rows)")
    print(f"wrote {SUBSET_IDS_JSON}")


if __name__ == "__main__":
    main()
