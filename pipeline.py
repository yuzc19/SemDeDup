import logging
import random

import numpy as np
import yaml
from clustering.clustering import compute_centroids
from clustering.sort_clusters import assign_and_sort_clusters
from datasets import Dataset
from extract_dedup_data import extract_pruned_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

confg_file = "clustering/configs/openclip/clustering_configs.yaml"
## -- Load kmeans clustering parameters from configs file
with open(confg_file, "r") as y_file:
    params = yaml.load(y_file, Loader=yaml.FullLoader)

## -- Fix the seed
SEED = params["seed"]
random.seed(SEED)
emb_memory_loc = params["emb_memory_loc"]
paths_memory_loc = params["paths_memory_loc"]
dataset_size = params["dataset_size"]
emb_size = params["emb_size"]
path_str_type = params["path_str_type"]

cc = False
asc = False
epd = False

if cc:
    dataset_path = "../data/fineweb/sample-350BT/train/0_embedding.jsonl"
    dataset = Dataset.from_json(dataset_path)
    embedding_matrix = np.array(dataset["__embedding"])
    dataset_size = len(embedding_matrix)
    # embedding_matrix = embedding_matrix[:dataset_size]

    # emb_array = np.memmap(
    #     emb_memory_loc,
    #     dtype="float32",
    #     mode="w+",
    #     shape=(dataset_size, emb_size),
    # )
    # emb_array[:] = embedding_matrix[:]
    # emb_array.flush()

    # emb_memory = np.memmap(
    #     emb_memory_loc,
    #     dtype="float32",
    #     mode="r",
    #     shape=(dataset_size, emb_size),
    # )

    path = [f"{i}" for i in range(dataset_size)]
    paths_array = np.memmap(
        paths_memory_loc,
        dtype=path_str_type,
        mode="w+",
        shape=(dataset_size,),
    )
    paths_array[:] = path[:]

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

if asc:
    dataset_path = "../data/fineweb/sample-350BT/train/0_embedding.jsonl"
    dataset = Dataset.from_json(dataset_path)
    embedding_matrix = np.array(dataset["__embedding"])

    # emb_memory = np.memmap(
    #     emb_memory_loc,
    #     dtype="float32",
    #     mode="r",
    #     shape=(dataset_size, emb_size),
    # )

    paths_memory = np.memmap(
        paths_memory_loc,
        dtype=path_str_type,
        mode="r",
        shape=(dataset_size,),
    )

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

if epd:
    eps = 0.6
    extract_pruned_data(
        "sorted_clusters",
        "semdedup/dataframes",
        eps,
        params["ncentroids"],
        "output.txt",
        retreive_kept_samples=True,
    )
