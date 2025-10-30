### Load query data
```bash
git clone https://github.com/amazon-science/esci-data.git # remember to enable lfs
python process_queries.py
```
> Huan: Filtering (union between queries and item subsets, languages) on queries needs to be done. I didn't save the code and I'll update this on Saturday :(.

### Constructing Database
Download the review and meta file from [Amazon Reviews'23](https://amazon-reviews-2023.github.io/main.html), unzip them, put them under `amz2023_raw` directory  

```bash
python process_amazon_reviews # this will process meta and review files into csv
sqlite3 amz.db < schema.sql
sqlite3 amz.db < load.txt
```
