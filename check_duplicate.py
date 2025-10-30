#!/usr/bin/env python3
import argparse, csv, glob, os, sys
from collections import defaultdict, Counter

def scan_files(key: str, require_key: bool, delimiter: str | None):
    file_list = [
        "amz2023-data/meta_All_Beauty.csv",
        "amz2023-data/meta_Amazon_Fashion.csv",
        "amz2023-data/meta_Appliances.csv",
    ]

    # Per-file stats
    per_file_counts = {}              # file -> total rows (with key)
    per_file_dups   = {}              # file -> {product_id: count (>1 only)}
    per_file_missing_key = {}         # file -> count of rows missing key

    # Global maps
    id_to_files = defaultdict(Counter)  # product_id -> Counter({file: count})
    total_rows = 0

    for path in file_list:
        counts = Counter()
        missing_key = 0
        rows_with_key = 0

        # Choose dialect/delimiter
        open_kwargs = dict(mode="r", encoding="utf-8", newline="")
        with open(path, **open_kwargs) as f:
            if delimiter is None:
                # auto sniff just the delimiter to be safe
                sample = f.read(4096)
                f.seek(0)
                try:
                    sniffer = csv.Sniffer()
                    dialect = sniffer.sniff(sample)
                    dialect.doublequote = True
                    reader = csv.DictReader(f, dialect=dialect)
                except Exception:
                    reader = csv.DictReader(f)
            else:
                reader = csv.DictReader(f, delimiter=delimiter)

            # sanity: ensure key in header
            if key not in (reader.fieldnames or []):
                msg = f"[WARN] {os.path.basename(path)}: column '{key}' not found in header: {reader.fieldnames}"
                if require_key:
                    print(msg, file=sys.stderr)
                    sys.exit(2)
                else:
                    print(msg, file=sys.stderr)

            for row in reader:
                total_rows += 1
                pid = (row.get(key) or "").strip() if row else ""
                if not pid:
                    missing_key += 1
                    continue
                rows_with_key += 1
                counts[pid] += 1
                id_to_files[pid][path] += 1

        per_file_counts[path] = rows_with_key
        per_file_missing_key[path] = missing_key
        per_file_dups[path] = {pid: c for pid, c in counts.items() if c > 1}

    return (file_list, per_file_counts, per_file_missing_key, per_file_dups, id_to_files, total_rows)

def write_report(report_path: str, id_to_files: dict, per_file_dups: dict):
    """
    Write a CSV with two sections:
      1) Cross-file duplicates: product_id appears in >=2 files
      2) Intra-file duplicates: product_id appears >1 times within the same file
    """
    with open(report_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section","product_id","files","counts_by_file"])

        # Cross-file dups
        for pid, files_counter in sorted(id_to_files.items(), key=lambda x: x[0]):
            if len(files_counter) >= 2:
                files = ";".join(os.path.basename(p) for p in files_counter.keys())
                counts = ";".join(f"{os.path.basename(p)}:{c}" for p, c in files_counter.items())
                w.writerow(["cross_file", pid, files, counts])

        # Intra-file dups
        for path, dups in per_file_dups.items():
            if not dups:
                continue
            for pid, cnt in sorted(dups.items()):
                w.writerow(["within_file", pid, os.path.basename(path), f"{cnt}"])

def main():
    ap = argparse.ArgumentParser(description="Check duplicate product_id across multiple products*.csv files.")
    ap.add_argument("--key", default="product_id", help="Column name to check (default: product_id)")
    ap.add_argument("--require-key", action="store_true", help="Fail if a file misses the key in header")
    ap.add_argument("--delimiter", default=None, help="CSV delimiter if not comma (default: auto-detect)")
    ap.add_argument("--report", default=None, help="Optional path to write a CSV report of duplicates")
    args = ap.parse_args()

    (files, per_file_counts, per_file_missing_key, per_file_dups, id_to_files, total_rows) = scan_files(
        key=args.key,
        require_key=args.require_key,
        delimiter=args.delimiter
    )

    n_files = len(files)
    n_unique_ids = len(id_to_files)
    n_cross_file_dup_ids = sum(1 for pid, c in id_to_files.items() if len(c) >= 2)
    n_within_file_dup_ids = sum(len(dups) for dups in per_file_dups.values())

    print("=== Duplicate Check Summary ===")
    print(f"Files scanned      : {n_files}")
    print(f"Total rows (all)   : {total_rows}")
    print(f"Unique {args.key}s : {n_unique_ids}")
    print(f"Cross-file duplicate {args.key}s : {n_cross_file_dup_ids}")
    print(f"Within-file duplicate {args.key}s: {n_within_file_dup_ids}")
    print()

    print("Per-file stats:")
    for path in files:
        print(f"- {os.path.basename(path)}")
        print(f"    rows with {args.key}: {per_file_counts.get(path,0)}")
        print(f"    rows missing  {args.key}: {per_file_missing_key.get(path,0)}")
        print(f"    duplicate {args.key}s within file: {len(per_file_dups.get(path,{}))}")

    # Show top 10 cross-file duplicates (if any)
    if n_cross_file_dup_ids:
        print("\nExamples of cross-file duplicates (up to 10):")
        shown = 0
        for pid, cnt in id_to_files.items():
            if len(cnt) >= 2:
                files_list = ", ".join(os.path.basename(p) for p in cnt.keys())
                counts_list = ", ".join(f"{os.path.basename(p)}:{c}" for p, c in cnt.items())
                print(f"  {args.key}={pid}  ->  files: [{files_list}]  counts: [{counts_list}]")
                shown += 1
                if shown >= 10:
                    break

    if args.report:
        write_report(args.report, id_to_files, per_file_dups)
        print(f"\nReport written to: {args.report}")

if __name__ == "__main__":
    main()
