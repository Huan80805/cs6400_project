## Build Database
1. Clone the query data:
```bash
git clone https://github.com/amazon-science/esci-data.git # remember to enable lfs
```
2. Download the review and meta file from [Amazon Reviews'23](https://amazon-reviews-2023.github.io/main.html), unzip them, put them under `amz2023_raw` directory
3. Pre-process Amazon Reviews'23
```bash
python 1_build_db/process_amazon_reviews.py --input_dir ./amz2023_raw --out_dir ./amz2023_processed # this will process meta and review files into csv
sqlite3 amz.db < 1_build_db/schema.sql
sqlite3 amz.db < 1_build_db/load_amz.txt
python 1_build_db/load_filters.py --db amz.db --json 1_build_db/filters_deduplicated.json
```
1. Filter queries and load into db  
> note: this will apply product subset filters and langauge filters (English).
```bash
python 1_build_db/process_queries.py --query_file esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet --db amz.db
sqlite3 amz.db < 1_build_db/load_queries.txt

```

## Running Evaluation
Download the product embedding (it takes a long while to generate them!): https://drive.google.com/file/d/1KPoeD8GW1MQpohbAI0lnNcLdxhhtt3hw/view?usp=sharing
Save it as ./embeddings.parquet
```bash
python main.py
```


### TODO:
- Separate filters from product table (otherwise, most of them are null vals)
- Amazon C4 queries