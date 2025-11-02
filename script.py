import pyarrow.parquet as pq
import argparse
import sys


def slice_parquet(input_path: str, output_path: str, n_rows: int):
    """
    Efficiently slices the top N rows from a Parquet file without
    loading the entire file into memory.
    """
    print(f"Opening input file: {input_path}")
    try:
        reader = pq.ParquetFile(input_path)
    except Exception as e:
        print(f"Error opening Parquet file: {e}")
        return

    # Use .schema_arrow to get the pyarrow.Schema
    arrow_schema = reader.schema_arrow
    print(f"Input schema:\n{arrow_schema}")

    # Open a writer with the correct arrow_schema
    try:
        # --- THIS IS THE FIX ---
        # Pass the pyarrow.Schema object (reader.schema_arrow)
        writer = pq.ParquetWriter(output_path, arrow_schema)
        # ---------------------
    except Exception as e:
        print(f"Error opening output file for writing: {e}")
        return

    rows_written = 0
    batch_size = 65536  # Read 65k rows at a time

    print(f"Starting to slice top {n_rows} rows...")
    try:
        for batch in reader.iter_batches(batch_size=batch_size):
            if rows_written + len(batch) >= n_rows:
                # This is the last batch we need
                rows_to_take = n_rows - rows_written
                final_batch = batch.slice(0, rows_to_take)
                writer.write_batch(final_batch)
                rows_written += len(final_batch)
                break  # We are done
            else:
                # Write the whole batch
                writer.write_batch(batch)
                rows_written += len(batch)

    except Exception as e:
        print(f"Error during read/write: {e}")
    finally:
        writer.close()
        print(f"Slice complete. Wrote {rows_written} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Slice a Parquet file.")
    parser.add_argument("-i", "--input", required=True, help="Input Parquet file path")
    parser.add_argument(
        "-o", "--output", required=True, help="Output Parquet file path"
    )
    parser.add_argument(
        "-n", "--rows", type=int, required=True, help="Number of rows to keep"
    )

    args = parser.parse_args()

    if args.input == args.output:
        print("Error: Input and output paths cannot be the same.", file=sys.stderr)
        sys.exit(1)

    slice_parquet(args.input, args.output, args.rows)


if __name__ == "__main__":
    main()
