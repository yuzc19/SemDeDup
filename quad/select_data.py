import os

import numpy as np
from litdata import optimize
from litdata.streaming import StreamingDataset, TokensLoader

dataset = StreamingDataset(
    input_dir="/data/users/zichunyu/data/fineweb/sample-350BT/train/0",
    item_loader=TokensLoader(block_size=2048 + 1),
)
dataset_size = len(dataset)
print(f">> Dataset size: {dataset_size}")

indices = np.load("cluster_info/selected_indices.npy").tolist()
optimize(
    fn=lambda index: dataset[index],
    inputs=indices,
    output_dir=f"/data/users/zichunyu/data/fineweb/sample-350BT/train/0/cluster/{len(indices)}",
    num_workers=(os.cpu_count() // 8),
    chunk_bytes="200MB",
)
