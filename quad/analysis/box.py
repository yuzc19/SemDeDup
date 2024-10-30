import os

import datasets
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Generate random influence scores for each cluster
np.random.seed(0)

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
high_influence_indices = np.argsort(average_reward)[::-1][:3]
high_influence_scores = []
for cluster_id in high_influence_indices:
    cluster_i = np.load(
        os.path.join(
            params["sorted_clusters_file_loc"],
            f"cluster_{cluster_id}.npy",
        )
    )
    indices = cluster_i[:, 0].astype("int32")
    print(indices.shape, np.std(metrics[indices]))
    high_influence_scores.append(-metrics[indices])

low_influence_indices = np.argsort(average_reward)[:3]
low_influence_scores = []
for cluster_id in low_influence_indices:
    cluster_i = np.load(
        os.path.join(
            params["sorted_clusters_file_loc"],
            f"cluster_{cluster_id}.npy",
        )
    )
    indices = cluster_i[:, 0].astype("int32")
    print(indices.shape, np.std(metrics[indices]))
    low_influence_scores.append(-metrics[indices])
# high_influence_scores = [
#     np.random.normal(0.01, 0.003, 100) for _ in range(3)
# ]  # C1, C2, C3
# low_influence_scores = [np.random.normal(0, 0.005, 100) for _ in range(3)]  # C4, C5, C6

# Combine all scores
data = high_influence_scores + low_influence_scores

# Plot the boxplot
fig, ax = plt.subplots()
plt.rcParams["figure.figsize"] = (10, 6)
box = ax.boxplot(data, notch=True, patch_artist=True, showfliers=True)

# Customize colors for high- and low-influence clusters
colors = ["teal", "lightseagreen", "turquoise", "lightcoral", "indianred", "brown"]
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)

# Set x-ticks and labels
ax.set_xticks(range(1, 7))
ax.set_xticklabels(["$C_1$", "$C_2$", "$C_3$", "$C_4$", "$C_5$", "$C_6$"])

# Set y-label and title
ax.set_ylabel("Influence Scores")
ax.set_title("Influence Scores in Different Clusters")

# Group the clusters for better readability
plt.text(2.0, -1.022, "low-influence clusters", ha="center", va="bottom", fontsize=12)
plt.text(5.0, -1.022, "high-influence clusters", ha="center", va="bottom", fontsize=12)

plt.tight_layout()
plt.savefig("box.png", bbox_inches="tight")
