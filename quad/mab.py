import os
import pathlib
import random

import datasets
import numpy as np
import torch
import yaml

alpha = 0.0001
batch = 50


def collect_reward(num_clusters):
    dataset = datasets.concatenate_datasets(
        [
            datasets.load_from_disk(
                f"/data/users/zichunyu/out/pythia-1b/fineweb/sample-350BT/train/0/10000-data_influence_model-flan-prediction/{i}"
            )
            for i in range(18)
        ]
    )
    metrics = np.array(dataset["prediction"]).reshape(-1)
    print(">> Metrics shape:", metrics.shape)

    cluster_average_reward = np.zeros(num_clusters)
    for cluster_id in range(num_clusters):
        cluster_i = np.load(
            os.path.join(
                params["sorted_clusters_file_loc"],
                f"cluster_{cluster_id}.npy",
            )
        )
        indices = cluster_i[:, 0].astype("int32")
        cluster_average_reward[cluster_id] = np.mean(metrics[indices])
    np.save(pathlib.Path("cluster_info", "average_reward.npy"), cluster_average_reward)


def mab(
    num_clusters,
    cluster_average_reward,
):
    sum_chose = 0
    cluster_chose_ratio = np.zeros(num_clusters)
    cluster_chose_time = np.zeros(num_clusters)
    cluster_ucb = cluster_average_reward.copy()
    for _ in range(1400):
        # 0.02 * 0.05 * 200
        # 1000 -> 0.2 selection ratio
        current_chose_num = 0
        current_chose = []
        for k in np.argsort(-cluster_ucb):
            if cluster_chose_ratio[k] < 1:
                cluster_chose_ratio[k] += 0.05
                cluster_chose_time[k] += 1
                current_chose_num += 1
                current_chose.append(k)
                if current_chose_num == batch:
                    break
        sum_chose += batch
        for k in range(num_clusters):
            ucb = alpha * np.sqrt(
                2 * (np.log(float(sum_chose))) / float(cluster_chose_time[k] + 1)
            )
            cluster_ucb[k] = cluster_average_reward[k] + ucb
    return cluster_chose_ratio


if __name__ == "__main__":
    confg_file = "clustering/configs/openclip/clustering_configs.yaml"
    with open(confg_file, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    SEED = params["seed"]
    random.seed(SEED)
    dataset_size = params["dataset_size"]
    num_clusters = params["ncentroids"]

    cluster_chose_ratio = mab(num_clusters, np.load("cluster_info/average_reward.npy"))
    np.save("cluster_info/cluster_chose_ratio.npy", cluster_chose_ratio)
    selected_indices = []
    for cluster_id in range(num_clusters):
        cluster_i = np.load(
            os.path.join(
                params["sorted_clusters_file_loc"],
                f"cluster_{cluster_id}.npy",
            )
        )
        indices = cluster_i[:, 0].astype("int32")
        random.shuffle(indices)
        selected_indices += indices[
            : int(cluster_chose_ratio[cluster_id] * len(indices))
        ].tolist()
    print(">> Selected indices shape:", len(selected_indices))
    np.save("selected_indices.npy", selected_indices)
