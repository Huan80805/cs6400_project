"""
Preprocesses raw Amazon review and metadata .jsonl files.

This script reads all .jsonl files from an input directory, processes them
into a standardized format, and writes the results to CSV files in an
output directory.

It expects two types of input files:
1.  Product Metadata: Files named 'meta_*.jsonl'
2.  Product Reviews:   All other '*.jsonl' files

For each input .jsonl file, a corresponding .csv file is created.

Example Usage:
    python process_amazon_reviews.py --input_dir ./amz2023_raw --out_dir ./amz2023_processed
"""

import argparse, os, json, csv, re, sqlite3, datetime
from typing import Any, Dict, Optional, Iterable
import glob

# Regular expression to extract the first valid number from a price string.
PRICE_RE = re.compile(r"[-+]?\d*\.?\d+")

# Set of strings to be interpreted as a 'True' boolean value.
TRUE_STR = {"true", "1", "y", "yes", "t"}

# Define the exact column order for the output metadata CSV.
META_COLUMNS = [
    "product_id", "parent_asin", "main_category", "product_title", "average_rating",
    "rating_number", "features_json", "product_description", "price",
    "images_json", "videos_json", "store", "categories_json", "details_json",
    "bought_together_json", "brand", "color", "product_locale"
]

# Define the exact column order for the output reviews CSV.
REVIEW_COLUMNS = [
    "parent_asin", "rating", "review_title", "review_text", "images_json",
    "user_id", "review_ts", "helpful_vote", "verified_purchase",
    "marketplace", "category"
]

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """
    Reads a JSONL (JSON Lines) file line by line, yielding each valid JSON object.
    Args:
        path: The file path to the .jsonl file.
    Yields:
        A dictionary for each valid JSON line in the file.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Tolerate occasional bad lines
                print(f"Warning: Skipping malformed JSON line in {path}")
                continue

def to_iso8601(ts: Any) -> Optional[str]:
    """
    Converts various timestamp formats into a standard ISO 8601 string.

    Accepts:
      - Already ISO-like strings ("2020-05-05 14:08:48.923")
      - Epoch seconds (int/float)
      - Other date-like strings

    Returns:
        A string in 'YYYY-MM-DD HH:MM:SS' format (UTC) or None if conversion fails.
    """
    if ts is None:
        return None

    # Handle epoch integers/floats
    if isinstance(ts, (int, float)):
        try:
            return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

    s = str(ts).strip()
    if not s:
        return None

    # Try common string formats
    for fmt in ("%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d"):
        try:
            dt = datetime.datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

    # Last resort: let fromisoformat try
    try:
        # Tidy up string for fromisoformat (remove 'Z', add timezone)
        dt = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
        # Remove timezone info to standardize
        return dt.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

def to_json_text(val: Any) -> Optional[str]:
    """
    Serializes a Python object (like a list or dict) into a JSON string.
    (TODO: just a lazy way to handle complex fields for now, may be subject to change)
    Args:
        val: The Python object to serialize.

    Returns:
        A JSON-formatted string, or the raw string value if serialization fails.
    """
    if val is None:
        return None
    try:
        # Standard serialization
        return json.dumps(val, ensure_ascii=False)
    except Exception:
        # Fallback: Sometimes the field is a string that looks like a Python
        # literal (e.g., "{'key': 'value'}"). Try a best-effort repair.
        s = str(val).strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            s_repaired = s.replace("'", '"')
            try:
                # Try to parse the repaired string
                parsed = json.loads(s_repaired)
                # Re-serialize it correctly
                return json.dumps(parsed, ensure_ascii=False)
            except Exception:
                return s  # Store raw string as last resort
        return s

def parse_price(val: Any) -> Optional[float]:
    """
    Extracts a float price from a string (e.g., "$19.99", "1,200.00").

    Args:
        val: The input value, typically a string.

    Returns:
        A float, or None if no price can be found.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip()
    if not s:
        return None

    # Remove thousands separators before regex search
    m = PRICE_RE.search(s.replace(",", ""))
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None

def to_int(val: Any) -> Optional[int]:
    """Safely converts a value to an integer, handling floats."""
    if val is None or val == "":
        return None
    try:
        return int(val)
    except Exception:
        try:
            # Handle cases where the number is a float string (e.g., "5.0")
            return int(float(val))
        except Exception:
            return None


def to_float(val: Any) -> Optional[float]:
    """Safely converts a value to a float."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except Exception:
        return None


def to_bool01(val: Any) -> Optional[int]:
    """
    Converts a "truthy" value (bool, str, int) to 1, 0, or None.

    Args:
        val: The input value.

    Returns:
        1 if True, 0 if False, None if indeterminate.
    """
    if val is None or val == "":
        return None
    if isinstance(val, bool):
        return 1 if val else 0

    s = str(val).strip().lower()
    if s in TRUE_STR:
        return 1
    if s in {"false", "0", "n", "no", "f"}:
        return 0
    return None

# Core Record Normalization
def normalize_meta(rec: Dict[str, Any], product_id: int) -> Dict[str, Any]:
    """
    Cleans and standardizes a single product metadata record.

    Args:
        rec: The raw dictionary from the 'meta_*.jsonl' file.
        product_id: A unique integer ID to assign to this product.

    Returns:
        A new dictionary with fields matching META_COLUMNS.
    """
    # The 'description' field can be a list of paragraphs or a single string.
    # We join lists into a single text block.
    desc = rec.get("description")
    product_description = None
    if isinstance(desc, list):
        product_description = "\n".join([str(x) for x in desc if x is not None])
    elif isinstance(desc, dict):
        # If it's a dict, store as JSON
        product_description = to_json_text(desc)
    elif desc is not None:
        product_description = str(desc)
    parent_asin = rec.get("parent_asin")

    out = {
        "product_id": product_id,
        "parent_asin": rec.get("parent_asin"),
        "main_category": rec.get("main_category"),
        "product_title": rec.get("title"),
        "average_rating": to_float(rec.get("average_rating")),
        "rating_number": to_int(rec.get("rating_number")),
        "features_json": to_json_text(rec.get("features")),
        "product_description": product_description,
        "price": parse_price(rec.get("price")),
        "images_json": to_json_text(rec.get("images")),
        "videos_json": to_json_text(rec.get("videos")),
        "store": rec.get("store"),
        "categories_json": to_json_text(rec.get("categories")),
        "details_json": to_json_text(rec.get("details")),
        "bought_together_json": to_json_text(rec.get("bought_together")),

        # Handle messy fields that sometimes use different capitalizations
        "brand": rec.get("brand") or rec.get("Brand"),
        "color": rec.get("color") or rec.get("Color"),
        "product_locale": rec.get("marketplace") or rec.get("locale") or rec.get("product_locale"),
    }
    return out

def normalize_review(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cleans and standardizes a single product review record.

    Args:
        rec: The raw dictionary from the '*.jsonl' review file.

    Returns:
        A new dictionary with fields matching REVIEW_COLUMNS.
    """
    out = {
        "parent_asin": rec.get("parent_asin"),
        "rating": to_float(rec.get("rating")),
        "review_title": rec.get("title"),
        "review_text": rec.get("text"),
        "images_json": to_json_text(rec.get("images")),

        # Handle multiple possible keys for the same data
        "user_id": rec.get("user_id") or rec.get("user") or rec.get("reviewerID"),
        "review_ts": to_iso8601(rec.get("timestamp") or rec.get("unixReviewTime") or rec.get("reviewTime")),
        "helpful_vote": to_int(rec.get("helpful_vote") or rec.get("helpful")),
        "verified_purchase": to_bool01(rec.get("verified_purchase") or rec.get("verified")),
        "marketplace": rec.get("marketplace") or rec.get("locale"),
        "category": rec.get("category") or rec.get("main_category"),
    }
    return out

def write_csv(rows: Iterable[Dict[str, Any]], path: str, columns: list) -> int:
    """
    Writes an iterable of dictionaries to a CSV file.

    Args:
        rows: An iterable (like a generator) of data rows (dictionaries).
        path: The output file path for the CSV.
        columns: A list of column names, defining the header and order.

    Returns:
        The total number of rows written to the file.
    """
    count = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            # Filter row to only include keys present in 'columns'
            w.writerow({k: r.get(k) for k in columns})
            count += 1
    return count

def main():
    """Main script entry point."""
    ap = argparse.ArgumentParser(description="Process Amazon .jsonl review/meta files into CSVs.")
    ap.add_argument("--out_dir", required=True, help="Directory to write output CSVs")
    ap.add_argument("--input_dir", default='amz2023-data', help="Directory to read input JSONL files")
    args = ap.parse_args()

    # Find all .jsonl files and sort them into meta vs. review
    files = glob.glob(os.path.join(args.input_dir, "*.jsonl"))
    meta_files = [f for f in files if os.path.split(f)[-1].startswith("meta_")]
    review_files = [f for f in files if not os.path.split(f)[-1].startswith("meta_")]

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Found {len(meta_files)} meta files and {len(review_files)} review files in '{args.input_dir}'")

    meta_csv_paths = []
    review_csv_paths = []

    # 1) Process metadata files
    print("\nProcessing metadata files...")
    # We need a unique, persistent product_id. We create one by
    # simple auto-incrementing integer.
    product_id_counter = 0
    for meta_file in meta_files:
        print(f"Processing {meta_file}...")
        
        # Create a generator that normalizes each record.
        # We pass `product_id_counter + i` to assign a unique ID to each row.
        meta_rows_generator = (
            normalize_meta(rec, product_id_counter + i)
            for i, rec in enumerate(read_jsonl(meta_file))
        )
        
        # Define the output CSV path
        csv_name = os.path.split(meta_file)[-1].replace(".jsonl", ".csv")
        meta_csv = os.path.join(args.out_dir, csv_name)

        # Write the CSV and get the count of rows written (for efficiency)
        rows_written = write_csv(meta_rows_generator, meta_csv, META_COLUMNS)
        
        print(f"- Meta CSV:    {meta_csv} ({rows_written} rows)")
        
        # Increment the global counter by the number of rows we just wrote
        product_id_counter += rows_written
        meta_csv_paths.append(meta_csv)

    # 2) Process review files
    print("\nProcessing review files...")
    for review_file in review_files:
        print(f"Processing {review_file}...")
        
        # Create a generator that normalizes each review
        review_rows_generator = (normalize_review(rec) for rec in read_jsonl(review_file))
        
        csv_name = os.path.split(review_file)[-1].replace(".jsonl", ".csv")
        reviews_csv = os.path.join(args.out_dir, csv_name)

        # Write the CSV and get the count
        rows_written = write_csv(review_rows_generator, reviews_csv, REVIEW_COLUMNS)
        
        print(f"- Reviews CSV: {reviews_csv} ({rows_written} rows)")
        review_csv_paths.append(reviews_csv)


    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
