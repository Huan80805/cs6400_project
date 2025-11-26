"""
Filter Selectivity Analysis for Product Search Queries.

This script analyzes how selective different product filters are by:
1. Loading query data and filtering to queries with valid products
2. Creating FTS5 index for text search
3. Generating filters (exact match, range, NER-based, JSON contains)
4. Measuring filter effectiveness and selectivity
5. Saving results to JSON files
"""

import json
import sqlite3
import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import argparse
from datasets import load_dataset

# Target selectivities for range filters (in percent)
TARGET_SELECTIVITIES_PCT = [0.1, 1.0, 10.0, 50.0]

# Filter column configuration
FILTER_COLUMNS_CONFIG = {
    'main_category': {'type': 'exact_match'},
    'product_title': {'type': 'NER_contains'},
    'store': {'type': 'exact_match'},
    'average_rating': {'type': 'range'},
    'rating_number': {'type': 'range'},
    'price': {'type': 'range'},
    'features_json': {'type': 'json_contains'},
    'details_json': {'type': 'json_contains'}
}


# --- Utility Functions ---

def is_value_empty(val):
    """Checks if a value is empty (None, NaN, '', '[]', '{}', 'null')."""
    if pd.isna(val):
        return True
    s_val = str(val).strip()
    if s_val == '' or s_val in ('[]', '{}') or s_val.lower() == 'null' or s_val.lower() == 'nan':
        return True
    return False


def _extract_recursive(obj, values_list):
    """
    Recursively navigates a JSON object (dict or list) and
    appends all "leaf" string values to the values_list.
    """
    if isinstance(obj, dict):
        for v in obj.values():
            _extract_recursive(v, values_list)
    elif isinstance(obj, list):
        for item in obj:
            _extract_recursive(item, values_list)
    elif obj is not None:
        val_str = str(obj).strip()
        if val_str:
            values_list.append(val_str)


def get_json_values(json_string):
    """
    Safely parses a JSON string and returns a flat list of ALL
    string values, including those in nested lists or dicts.
    """
    if is_value_empty(json_string):
        return []
    
    values_to_return = []
    try:
        obj = json.loads(json_string)
        _extract_recursive(obj, values_to_return)
    except (json.JSONDecodeError, TypeError):
        print(f'Warning, bad JSON: {json_string}')
        return []
    
    return values_to_return


def extract_entities(text, nlp):
    """Extract named entities from a string using spaCy."""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ['CARDINAL', 'ORDINAL']:
            continue
        entities.append(ent.text)
    if len(entities) == 0:
        return [text]
    return entities


def get_filter_stats(cursor, where_clause, params, asin_to_check, total_products_in_db, use_fts=False):
    """
    Checks if a filter is effective (finds its own ASIN) and, if so,
    returns its selectivity (match count and percentage).
    
    Returns: A dict {'match_count': int, 'match_percentage': float} or None
    """
    db_table = "products_fts" if use_fts else "products"
    
    # Effectiveness Check
    check_sql = f"SELECT 1 FROM {db_table} WHERE ({where_clause}) AND parent_asin = ? LIMIT 1"
    check_params = params + (asin_to_check,)
    if cursor.execute(check_sql, check_params).fetchone() is None:
        print(f"Ineffective filter detected with SQL query: {check_sql}, params: {check_params}")
        return None
    
    # Selectivity Check
    count_sql = f"SELECT COUNT(*) FROM {db_table} WHERE ({where_clause})"
    count = cursor.execute(count_sql, params).fetchone()[0]
    percentage = round((count / total_products_in_db) * 100, 3)
    if percentage == 0:
        print(f"Ineffective filter detected (0% selectivity) with SQL query: {count_sql}, params: {params}")
        return None
    
    return {
        'match_count': count,
        'match_percentage': percentage
    }


# --- Data Loading ---

def prefetch_range_distributions(conn, cursor, filter_config):
    """Pre-fetch sorted values for range columns."""
    print("Pre-fetching range column data...")
    range_columns = [col for col, conf in filter_config.items() if conf['type'] == 'range']
    distributions = {}
    
    total_products = cursor.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    if total_products == 0:
        raise Exception("No products in database.")
    
    for col in range_columns:
        print(f"  Fetching '{col}' values...")
        df_dist = pd.read_sql_query(f"SELECT {col} FROM products WHERE {col} IS NOT NULL", conn)
        series_dist = pd.to_numeric(df_dist[col], errors='coerce').dropna()
        distributions[col] = np.sort(series_dist.values)
        print(f"  -> Fetched {len(distributions[col])} values for '{col}'.")
    
    return distributions, total_products


def create_fts_table(cursor, conn):
    """Create and populate FTS5 virtual table for text search."""
    print("Creating FTS5 virtual table 'products_fts'...")
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS products_fts USING fts5(
            parent_asin, 
            features_json, 
            details_json,
            product_title
        );
    """)
    
    print("Clearing FTS table...")
    cursor.execute("DELETE FROM products_fts")
    
    cursor.execute("""
        INSERT INTO products_fts (parent_asin, features_json, details_json, product_title)
        SELECT parent_asin, features_json, details_json, product_title
        FROM products 
        WHERE parent_asin IS NOT NULL;
    """)
    conn.commit()
    print("FTS table created and populated.")


# --- Filter Processing ---

def process_exact_match(row, col, cursor, asin, total_products):
    """Process exact match filter."""
    value = row.get(col)
    if is_value_empty(value):
        return []
    
    where_clause = f"LOWER({col}) = LOWER(?)"
    params = (str(value),)
    
    stats = get_filter_stats(cursor, where_clause, params, asin, total_products)
    if stats:
        return [{
            'filter_column': col,
            'filter_value': value,
            'total_in_db': total_products,
            **stats
        }]
    return []


def process_ner_contains(row, col, cursor, asin, total_products, nlp):
    """Process NER-based contains filter using FTS."""
    text = row.get(col)
    if is_value_empty(text):
        return []
    
    filters = []
    entities = extract_entities(text, nlp)
    
    for entity in entities:
        where_clause = f"{col} MATCH ?"
        escaped_val = entity.replace('"', '""')
        params = ('"' + escaped_val + '"',)
        
        stats = get_filter_stats(cursor, where_clause, params, asin, total_products, use_fts=True)
        if stats:
            filters.append({
                'filter_column': col,
                'filter_value': entity,
                'total_in_db': total_products,
                **stats
            })
    return filters


def process_range_filter(row, col, cursor, asin, total_products, distributions, target_selectivities):
    """Process range filter with target selectivities."""
    value = row.get(col)
    if is_value_empty(value):
        return []
    
    num_value = pd.to_numeric(value, errors='coerce')
    if pd.isna(num_value):
        return []
    
    sorted_values = distributions.get(col)
    if sorted_values is None or len(sorted_values) == 0:
        return []
    
    current_value_idx = np.searchsorted(sorted_values, num_value)
    total_values_for_col = len(sorted_values)
    
    filters = []
    for target_pct in target_selectivities:
        target_count = max(1, int((target_pct / 100.0) * total_values_for_col))
        half_count = target_count // 4
        
        lower_idx = max(0, current_value_idx - half_count)
        upper_idx = min(total_values_for_col - 1, current_value_idx + half_count)
        
        lower_bound_val = sorted_values[lower_idx]
        upper_bound_val = sorted_values[upper_idx]
        
        delta = max(num_value - lower_bound_val, upper_bound_val - num_value)
        min_val = max(0, num_value - delta)
        max_val = num_value + delta
        
        if col == 'rating_number':
            min_val, max_val = int(min_val), int(max_val)
        
        where_clause = f"{col} BETWEEN ? AND ?"
        params = (min_val, max_val)
        
        stats = get_filter_stats(cursor, where_clause, params, asin, total_products)
        if stats:
            filters.append({
                'filter_column': col,
                'filter_value': (min_val, max_val),
                'total_in_db': total_products,
                **stats
            })
    return filters


def process_json_contains(row, col, cursor, asin, total_products, nlp):
    """Process JSON contains filter using FTS."""
    value = row.get(col)
    values_to_check = get_json_values(value)
    if not values_to_check:
        return []
    
    filters = []
    for val in values_to_check:
        where_clause = f"{col} MATCH ?"
        entities = extract_entities(val, nlp)
        
        for entity in entities:
            escaped_val = entity.replace('"', '""')
            params = ('"' + escaped_val + '"',)
            
            stats = get_filter_stats(cursor, where_clause, params, asin, total_products, use_fts=True)
            if stats:
                filters.append({
                    'filter_column': col,
                    'filter_value': entity,
                    'total_in_db': total_products,
                    **stats
                })
    return filters


def analyze_product_filters(row, cursor, total_products, distributions, nlp):
    """Analyze all filters for a single product."""
    asin = row['parent_asin']
    output_record = row.to_dict()

    output_record['filters'] = []
    
    for col, config in FILTER_COLUMNS_CONFIG.items():
        filter_type = config['type']
        
        if filter_type == 'exact_match':
            output_record['filters'].extend(
                process_exact_match(row, col, cursor, asin, total_products)
            )
        elif filter_type == 'NER_contains':
            output_record['filters'].extend(
                process_ner_contains(row, col, cursor, asin, total_products, nlp)
            )
        elif filter_type == 'range':
            output_record['filters'].extend(
                process_range_filter(row, col, cursor, asin, total_products, 
                                    distributions, TARGET_SELECTIVITIES_PCT)
            )
        elif filter_type == 'json_contains':
            output_record['filters'].extend(
                process_json_contains(row, col, cursor, asin, total_products, nlp)
            )
    
    return output_record


def deduplicate_filters(all_outputs):
    """Deduplicate filters and print threshold statistics."""
    thresholds = [0.01, 0.1, 1.0, 10, 50, 70]
    
    for output in all_outputs:
        filters = output['filters']
        unique_filters = {}
        
        for f in filters:
            key = (f['filter_column'], f['match_count'], f['match_percentage'])
            if key not in unique_filters:
                unique_filters[key] = f
        
        filters = list(unique_filters.values())
        threshold_filters = {}
        for t in range(len(thresholds) - 1):
            interval = (thresholds[t], thresholds[t + 1])
            threshold_filters[interval] = len([
                f for f in filters 
                if interval[0] < f['match_percentage'] <= interval[1]
            ])
        
        print(threshold_filters)
        output['filters'] = filters
    
    return all_outputs


# --- Main Entry Point ---

def main(args):
    print("=" * 60)
    print("Filter Selectivity Analysis")
    print("=" * 60)
    
    # Connect to database
    print(f"\nConnecting to database: {args.db}...")
    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()
    
    try:
        # Load spaCy model
        print("\nLoading spaCy model...")
        nlp = spacy.load("en_core_web_sm")
        
        # Get valid product ASINs
        cursor.execute("SELECT DISTINCT parent_asin FROM products")
        rows = cursor.fetchall()
        valid_product_asins= {row[0] for row in rows if row[0]}
        print(f"Loaded {len(valid_product_asins)} unique product ASINs from database.")
        
        # Load and filter queries
        if args.query_csv == 'McAuley-Lab/Amazon-C4':
            dataset = load_dataset('McAuley-Lab/Amazon-C4')['test']
            df_queries = dataset.to_pandas()
            df_queries = df_queries.rename(columns={'item_id': 'product_id'})

        else: df_queries = pd.read_csv(args.query_csv)

        print(f"Loaded {len(df_queries)} queries.")
        initial_rows = len(df_queries)
        df_queries = df_queries[df_queries['product_id'].isin(valid_product_asins)].copy()
        print(f"Filter (Product ID in DB): {initial_rows} -> {len(df_queries)} rows")
        
        # Get products of interest
        product_ids = df_queries['product_id'].unique()
        placeholders = ','.join(['?'] * len(product_ids))
        sql_query = f"SELECT * FROM products WHERE parent_asin IN ({placeholders})"
        df_products = pd.read_sql_query(sql_query, conn, params=list(product_ids))
        print(f"Selected {len(df_products)} matching products from database.")

        
        # Pre-fetch range distributions
        distributions, total_products = prefetch_range_distributions(
            conn, cursor, FILTER_COLUMNS_CONFIG
        )
        
        # Create FTS table
        create_fts_table(cursor, conn)
        
        # Analyze filters for each product
        print(f"\nAnalyzing filters for {len(df_products)} products...")
        all_outputs = []
        
        total_filters_generated = 0
        pbar = tqdm(df_products.iterrows(), total=len(df_products))
        for i, (_, row) in enumerate(pbar):
            output = analyze_product_filters(
                row, cursor, total_products, distributions, nlp
            )
            # don't save product_id in output, avoiding confusion
            if 'product_id' in output:
                del output['product_id']
            all_outputs.append(output)
            
            # Update progress bar with average filters per product
            total_filters_generated += len(output['filters'])
            avg_filters = total_filters_generated / (i + 1)
            pbar.set_postfix({"avg_filters": f"{avg_filters:.2f}"})
        
        # Save all outputs
        print(f"\nSaving results to {args.output_all}...")
        with open(args.output_all, 'w') as f:
            json.dump(all_outputs, f, indent=2)
        
        # Deduplicate and save
        print(f"Deduplicating and saving to {args.output_deduplicated}...")
        all_outputs = deduplicate_filters(all_outputs)
        with open(args.output_deduplicated, 'w') as f:
            json.dump(all_outputs, f, indent=2)
        
        print("\nDone!")
        
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter Selectivity Analysis")
    parser.add_argument('--db', type=str, default='amz.db', help='Path to SQLite database file')
    parser.add_argument('--query_csv', type=str, default='esci-data/shopping_queries_dataset_small.csv', help='Path to query CSV file')
    parser.add_argument('--output_all', type=str, default='1_build_db/amz_c4_filters_all.json', help='Output file for all filters')
    parser.add_argument('--output_deduplicated', type=str, default='1_build_db/amz_c4_filters_deduplicated.json', help='Output file for deduplicated filters')
    args = parser.parse_args()
    main(args)