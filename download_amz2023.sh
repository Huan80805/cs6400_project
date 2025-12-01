#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="amz2023_raw"
mkdir -p "$TARGET_DIR"

URLS=(
  "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz"
  "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_All_Beauty.jsonl.gz"
  "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Amazon_Fashion.jsonl.gz"
  "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Amazon_Fashion.jsonl.gz"
  "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Appliances.jsonl.gz"
  "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Appliances.jsonl.gz"
)

for url in "${URLS[@]}"; do
  fname="$(basename "$url")"
  echo "Downloading $fname ..."
  curl -L -o "$TARGET_DIR/$fname" "$url"

  echo "Unzipping $fname ..."
  gunzip -f "$TARGET_DIR/$fname"
done

echo "Done. Files are in: $TARGET_DIR"
