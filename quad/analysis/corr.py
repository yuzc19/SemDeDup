import os

import datasets
import numpy as np
import yaml
from scipy.stats import spearmanr

average_reward = np.load("cluster_info/average_reward.npy")
print(average_reward.shape)
cluster_chose_ratio = np.load("cluster_info/cluster_chose_ratio.npy")
print(cluster_chose_ratio.shape)


dataset = datasets.concatenate_datasets(
    [
        datasets.load_from_disk(
            f"/data/users/zichunyu/out/pythia-1b/fineweb/sample-350BT/train/0/10000-data_influence_model-flan-prediction/{i}"
        )
        for i in range(18)
    ]
)
metrics = np.array(dataset["prediction"]).reshape(-1)
new_metrics = metrics.copy()
print(">> Metrics shape:", metrics.shape)

confg_file = "clustering/configs/openclip/clustering_configs.yaml"
with open(confg_file, "r") as y_file:
    params = yaml.load(y_file, Loader=yaml.FullLoader)
num_clusters = params["ncentroids"]
for cluster_id in range(num_clusters):
    cluster_i = np.load(
        os.path.join(
            params["sorted_clusters_file_loc"],
            f"cluster_{cluster_id}.npy",
        )
    )
    indices = cluster_i[:, 0].astype("int32")
    new_metrics[indices] = average_reward[cluster_id]

print(spearmanr(average_reward, cluster_chose_ratio))

print(spearmanr(metrics[:10000], new_metrics[:10000]))
print(spearmanr(metrics[:100000], new_metrics[:100000]))
