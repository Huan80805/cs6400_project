### Load query data
```bash
git clone https://github.com/amazon-science/esci-data.git # remember to enable lfs
python process_queries.py
```
This will apply product subset filters and langauge filters (English).

### Constructing Database
Download the review and meta file from [Amazon Reviews'23](https://amazon-reviews-2023.github.io/main.html), unzip them, put them under `amz2023_raw` directory  

```bash
python process_amazon_reviews # this will process meta and review files into csv
sqlite3 amz.db < schema.sql
sqlite3 amz.db < load.txt
```
