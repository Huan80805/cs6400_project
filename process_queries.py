import os
import pandas as pd
import sqlite3
from langdetect import detect, LangDetectException

df_examples = pd.read_parquet('esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet')
df_queries = df_examples[df_examples["small_version"] == 1]
DATABASE_FILE = 'amz.db'

if not os.path.exists(DATABASE_FILE):
    raise FileNotFoundError(f"Database file '{DATABASE_FILE}' not found. Need to build Database before proceessing queries.")

print("\nDataFrame Info:")
print(df_queries.info())


# Create a connection
try:
    conn = sqlite3.connect(DATABASE_FILE)
    print(f"Successfully connected to {DATABASE_FILE}")
except sqlite3.Error as e:
    print(f"Error connecting to database: {e}")


# Get Valid Product IDs from Database
valid_product_asins = set()
cursor = conn.cursor()
cursor.execute("SELECT DISTINCT parent_asin FROM products")

rows = cursor.fetchall()
for row in rows:
    if row[0]: # Ensure it's not None
        valid_product_asins.add(row[0])
        
print(f"Loaded {len(valid_product_asins)} unique product ASINs from database.")

# Filter 1: Keep rows where 'product_id' is in our set of valid ASINs
initial_rows = len(df_queries)
df_filtered_1 = df_queries[df_queries['product_id'].isin(valid_product_asins)].copy()
filtered_rows_1 = len(df_filtered_1)

print(f"Filter 1 (Product ID):")
print(f"Original rows: {initial_rows}")
print(f"Rows after filtering: {filtered_rows_1}")
print(f"Rows removed: {initial_rows - filtered_rows_1}")


def is_english(text):
    """Returns True if text is detected as English, False otherwise."""
    if not isinstance(text, str) or not text.strip():
        return False  # Handle empty strings or non-string types
    try:
        # Detect the language
        if detect(text) == 'en':
            return True
        else:
            return False
    except LangDetectException:
        # Failed to detect (e.g., text is just punctuation or numbers)
        return False
    except Exception as e:
        print(f"Error detecting language for '{text}': {e}")
        return False
    

initial_rows_2 = len(df_filtered_1)

# Apply the language detection function to the 'query' column
df_filtered_1['is_english'] = df_filtered_1['query'].apply(is_english)

# Filter 2: Keep rows where 'is_english' is True
df_final_filtered = df_filtered_1[df_filtered_1['is_english'] == True].copy()

# We can drop the temporary 'is_english' column
df_final_filtered = df_final_filtered.drop(columns=['is_english'])

filtered_rows_2 = len(df_final_filtered)

print(f"Filter 2 (Language):")
print(f"Rows before language filter: {initial_rows_2}")
print(f"Rows after language filter: {filtered_rows_2}")
print(f"Rows removed: {initial_rows_2 - filtered_rows_2}")

# Save the final filtered DataFrame to a new csv file
output_file = 'esci-data/shopping_queries_dataset/shopping_queries_dataset_filtered.csv'
df_final_filtered.to_csv(output_file, index=False)
print(f"Final filtered dataset saved to {output_file}")

