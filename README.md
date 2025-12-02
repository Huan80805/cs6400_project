# Implementation of Report Sections

## Section 2.2 (Data Pipeline):

- `./download_amz2023.sh`: Shell script for downloading the Amazon Reviews'23 Dataset
- `./1_build_db/schema.sql`: SQLite database schema definition
- `./db.py`: Main interface for itneracting with SQLite
- `./1_build_db/process_amazon_reviews.py`: Pre-processing logic for products and reviews data, creates csv files ready for SQLite import into `products` and `reviews` tables
- `./1_build_db/process_queries.py`: Pre-processing logic for ESCI and Amazon-C4 queries, creates csv files ready for SQLite import into `esci_queries` and `amz_c4_queries` tables
- `./encoder.py`: Main BLAIR embedding implementation
- `embeddings_pipeline.py`: Create product embeddings using the titles and descriptions of products, and save the generated `.parquet` file
- `query_vector_store.py`: Create query embeddings using the `esci_queries` and `amz_c4_queries` tables, and save the generated `.parquet` file

## Section 2.3 (Structured Filter Generation):

- `./1_build_db/gen_filters.py`: Main filter extraction logic used in hybrid search

## Section 2.4 (Indexes and Roaring Bitmaps):

- `./index/base.py`: Abstract class for Faiss index classes
- `./index/flat_l2.py`: Wrapper class for the Faiss IndexFlatL2 class
- `./index/ivfpq.py`: Wrapper class for the Faiss IndexIVFPQ class and implementation for index building and training
- `./bitmap/build_bitmaps.py`: Implementation for creating the global Roaring bitmap used in hybrid search
- `./bitmap/roaring_index.py`: Wrapper class containing the logic for fetching product_ids given a structured filter

## Section 2.5 (Hybrid Query Algorithms)

- `./search/base.py`: Abstract class for different hybrid search implementations
- `./search/flat_prefilter.py`: Implementation for pre-filter using SQL + Faiss FlatL2 index
- `./search/flat_prefilter_roaring.py`: Implementation for pre-filter using Roaring bitmap + Faiss FlatL2 index
- `./search/ivfpq_prefilter.py`: Implementation for pre-filter using SQL + Faiss IVFPQ index
- `./search/ivfpq_prefilter_roaring.py`: Implementation for pre-filter using Roaring bitmap + Faiss IVFPQ index
- `./search/flat_postfilter.py`: Implementation for post-filter using Faiss FlatL2 index + SQL
- `./search/flat_postfilter_roaring.py`: Implementation for post-filter using Faiss FlatL2 index + Roaring bitmap
- `./search/ivfpq_postfilter.py`: Implementation for post-filter using Faiss IVFPQ index + SQL
- `./search/ivfpq_postfilter_roaring.py`: Implementation for post-filter using Faiss IVFPQ index + Roaring bitmap

## Section 3 (Experimental Setup)

- `./main.py`: Main evaluation loop for different search classes outlined above

# Setup

## Build Database

To simplify setup, we can directly download the sqlite snapshot using

```bash
curl https://cs6400.s3.ap-northeast-1.amazonaws.com/amz.db --out amz.db
```

Or, to generate the database from scratch, we can follow the steps below:

1. Clone the ESCI query data:

```bash
git clone https://github.com/amazon-science/esci-data.git # remember to enable lfs
```

2. Download the review and meta file from [Amazon Reviews'23](https://amazon-reviews-2023.github.io/main.html) by running the following:

```bash
chmod +x download_amz2023.sh
./download_amz2023.sh
```

The downloaded files are stored under the directory ./amz2023_raw.

3. Download the extracted structured filters using

```bash
curl https://cs6400.s3.ap-northeast-1.amazonaws.com/esci_filters.json --out ./1_build_db/esci_filters.json
curl https://cs6400.s3.ap-northeast-1.amazonaws.com/amz_c4_filters.json --out ./1_build_db/amz_c4_filters.json
```

These filters are extracted from the ESCI and Amazon C4 datasets and matched to a calculated selectivity level (`match_percentage`). These levels will be used for hybrid query generation in our evaluation. The downloaded files are stored under the directory ./1_build_db.

4. Make sure the necessary python packages are installed

```bash
pip install -r requirements.txt
```

5. Pre-process Amazon Reviews'23, create the .csv data files, and store under the directory `./amz2023_processed`. Then, create the main sqlite database and populate the `products` and `reviews` tables.

```bash
python 1_build_db/process_amazon_reviews.py --input_dir ./amz2023_raw --out_dir ./amz2023_processed
sqlite3 amz.db < 1_build_db/schema.sql
sqlite3 amz.db < 1_build_db/load_amz.txt
```

6. Pre-process the queries data from ESCI and Amazon-C4, create shopping_queries_dataset_small.csv and amz_c4_queries.csv, and store under the directories `./esci-data` and `./1_build_db` respectively. This will populate the `esci_queries` and `amz_c4_queries` tables with only rows whose `product_id` exists as `parent_asin` in the products table, and whose query is in English.

```bash
python 1_build_db/process_queries.py --query_file esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet --db amz.db
python 1_build_db/process_queries.py --query_file McAuley-Lab/Amazon-C4 --db amz.db
sqlite3 amz.db < 1_build_db/load_queries.txt
```

7. Populate the `esci_filters` and `amz_c4_filters` tables with the structured filters to be used during hybrid search. We will only populate rows whose `product_id` exists as `parent_asin` in the products table.

```bash
python 1_build_db/load_filters.py --db amz.db --json 1_build_db/esci_filters.json --table esci_filters
python 1_build_db/load_filters.py --db amz.db --json 1_build_db/amz_c4_filters.json --table amz_c4_filters
```

## Build roaring bitmaps based on the stored structured filters

If using ESCI dataset, directly download bitmaps_esci.pkl using

```bash
curl https://cs6400.s3.ap-northeast-1.amazonaws.com/bitmaps_esci.pkl --out bitmaps_esci.pkl
```

Or generate from the most up-to-date code using

```bash
python -m bitmap.build_bitmaps --db amz.db --filters_json ./1_build_db/esci_filters.json --out bitmaps_esci.pkl
```

Note that the command above takes ~24mins on a 24-core, 160GB RAM PACE cluster.

If using Amazon-C4 dataset, directly download bitmaps_amz_c4.pkl using

```bash
curl https://cs6400.s3.ap-northeast-1.amazonaws.com/bitmaps_amz_c4.pkl --out bitmaps_amz_c4.pkl
```

Or generate from the most up-to-date code using

```bash
python -m bitmap.build_bitmaps --db amz.db --filters_json ./1_build_db/amz_c4_filters.json --out bitmaps_amz_c4.pkl
```

This command takes ~4.5mins on a 24-core, 160GB RAM PACE cluster.

## Create the product embeddings using the BLAIR model

Directly download embeddings.parquet using

```bash
curl https://cs6400.s3.ap-northeast-1.amazonaws.com/embeddings.parquet --out embeddings.parquet
```

Embedding generation using embeddings_pipeline.py takes hours, but here is the command:

```bash
python embeddings_pipeline.py --db ./amz.db --out_dir ./embedding_shards --categories Appliances All\ Beauty AMAZON\ FASHION --parquet embeddings.parquet
```

## Running Evaluation

```bash
python main.py --dataset [esci, amazon_c4]
```
