"""
esci
"""

import pandas as pd
df_examples = pd.read_parquet('esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet')
# load larger version
df_task_1 = df_examples[df_examples["small_version"] == 1]
print(len(df_task_1))
# TODO: make sure the queries' corresponding items are in the item subsets
# TODO: filter langauges as specified in the original paper

df_task_1.to_csv('esci-data/shopping_queries_dataset_small.csv', index=False)