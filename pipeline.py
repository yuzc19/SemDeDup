import logging
import random

import numpy as np
import yaml
from clustering.clustering import compute_centroids
from datasets import load_from_disk, concatenate_datasets
from clustering.sort_clusters import assign_and_sort_clusters
from datasets import Dataset
from extract_dedup_data import extract_pruned_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

confg_file = "clustering/configs/openclip/clustering_configs.yaml"
## -- Load kmeans clustering parameters from configs file
print("Loading configuration file...")
with open(confg_file, "r") as y_file:
    params = yaml.load(y_file, Loader=yaml.FullLoader)
print("Configuration file loaded.")

## -- Fix the seed
SEED = params["seed"]
random.seed(SEED)
emb_memory_loc = params["emb_memory_loc"]
paths_memory_loc = params["paths_memory_loc"]
dataset_size = params["dataset_size"]
emb_size = params["emb_size"]
path_str_type = params["path_str_type"]

cc = True
asc = True
epd = False

if cc:
    print("Loading datasets...")
    dataset = concatenate_datasets([load_from_disk(f"/data/datasets/hf_cache/data/fineweb/sample-350BT/train_bge_micro_embeddings/{i}") for i in range(8)])
    #test
    # dataset = concatenate_datasets([load_from_disk(f"/data/datasets/hf_cache/data/fineweb/sample-350BT/train_bge_micro_embeddings/{i}") for i in range(1)])
    # dataset = dataset.select(range(1000)) # test
    print("Datasets loaded.")

    embedding_matrix = np.array(dataset["__embedding"], dtype=np.float32)
    dataset_size = len(embedding_matrix)
    print(f"Embedding matrix created with size: {dataset_size}")

    path = [f"{i}" for i in range(dataset_size)]
    paths_array = np.memmap(
        paths_memory_loc,
        dtype=path_str_type,
        mode="w+",
        shape=(dataset_size,),
    )
    paths_array[:] = path[:]
    print("Paths array created.")

    print("Computing centroids...")
    compute_centroids(
        data=embedding_matrix,
        ncentroids=params["ncentroids"],
        niter=params["niter"],
        seed=params["seed"],
        Kmeans_with_cos_dist=params["Kmeans_with_cos_dist"],
        save_folder=params["save_folder"],
        logger=logger,
        verbose=True,
    )
    print("Centroids computed.")

if asc:
    print("Loading datasets...")
    dataset = concatenate_datasets([load_from_disk(f"/data/datasets/hf_cache/data/fineweb/sample-350BT/train_bge_micro_embeddings/{i}") for i in range(8)])
    print("Datasets loaded.")
    embedding_matrix = np.array(dataset["__embedding"], dtype=np.float32)
    print("Dataset loaded for sorting clusters.")

    paths_memory = np.memmap(
        paths_memory_loc,
        dtype=path_str_type,
        mode="r",
        shape=(dataset_size,),
    )
    print("Paths memory loaded.")

    print("Assigning and sorting clusters...")
    assign_and_sort_clusters(
        data=embedding_matrix,
        paths_list=paths_memory,
        sim_metric=params["sim_metric"],
        keep_hard=params["keep_hard"],
        kmeans_with_cos_dist=params["Kmeans_with_cos_dist"],
        save_folder=params["save_folder"],
        sorted_clusters_file_loc=params["sorted_clusters_file_loc"],
        cluster_ids=range(0, params["ncentroids"]),
        logger=logger,
    )
    print("Clusters assigned and sorted.")

# if epd:
#     eps = 0.6
#     extract_pruned_data(
#         "sorted_clusters",
#         "semdedup/dataframes",
#         eps,
#         params["ncentroids"],
#         "output.txt",
#         retreive_kept_samples=True,
#     )
