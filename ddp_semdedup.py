import math
import os
import pickle
import pprint
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from tqdm import tqdm

# Add any missing imports or functions, such as init_memmap_embs, here


def init_memmap_embs(
    embs_memory_loc: str,
    dataset_size: int,
    emd_size: int = 512,
    dtype: str = "float32",
) -> np.memmap:
    """
    Initializes a memory-mapped NumPy array to read embeddings of examples.

    Args:
        embs_memory_loc (str): Path to the memory-mapped file.
        dataset_size (int): Size of the dataset.
        emd_size (int): Dimensionality of the embeddings.
        dtype (str): Data type of the embeddings.

    Returns:
        np.memmap: A memory-mapped NumPy array.
    """
    embs = np.memmap(
        embs_memory_loc,
        dtype=dtype,
        mode="r",
        shape=(dataset_size, emd_size),
    )
    return embs


class SemDeDupJob:
    """
    - Each job will run SemDeDup on a number of clusters and save a dataframe indicating which examples to keep from each cluster.
    - Parallelize job_start_cluster across processes.
    - Process more than one cluster per job => run multiple tasks inside each job.
    - Already processed clusters get skipped internally.
    """

    def __init__(self, args, job_start_cluster: int):
        self.args = args
        self.job_start_cluster = job_start_cluster
        random.seed(args.seed)

    def _contains_duplicates(self, arr):
        return len(np.unique(arr)) != len(arr)

    def semdedup(self, cluster, cluster_reps, device):
        st = time.time()
        # Compute pairwise cosine similarity between cluster items
        cluster_reps = cluster_reps.to(device)
        print(cluster_reps.shape)
        pair_w_sim_matrix = cluster_reps @ cluster_reps.T
        del cluster_reps
        pair_w_sim_matrix.fill_diagonal_(0.0)
        assert pair_w_sim_matrix.shape[0] == pair_w_sim_matrix.shape[1]

        # Get paths to cluster images
        image_urls = cluster[:, 0]

        # Ensure all the paths are unique
        # assert not self._contains_duplicates(image_urls)

        # Use upper triangular matrix to ignore self-similarity and duplicate computations
        triu_sim_mat = torch.triu(pair_w_sim_matrix, diagonal=1)

        # If the max similarity between one example and any other example is > 1 - eps, remove this example
        M = torch.max(triu_sim_mat, dim=0)[0].cpu()
        print(f"Step time: {time.time()-st}(s)")

        return M

    def _process_shard(self, start_cluster: int, end_cluster: int):
        st = time.time()

        device = torch.device(f"cuda:{dist.get_rank() % torch.cuda.device_count()}")

        embs = init_memmap_embs(
            self.args.embs_memory_loc,
            self.args.dataset_size,
            self.args.emd_size,
        )

        step_time = []

        for cluster_id in tqdm(range(start_cluster, end_cluster)):
            step_st = time.time()

            df_file_loc = os.path.join(
                self.args.save_loc, f"dataframes/cluster_{cluster_id}.pkl"
            )

            if os.path.exists(df_file_loc):
                print(f"{df_file_loc} exists, moving on")
                continue

            # Load cluster representations
            cluster_i = np.load(
                os.path.join(
                    self.args.sorted_clusters_path, f"cluster_{cluster_id}.npy"
                )
            )

            # Store cluster size
            cluster_size = cluster_i.shape[0]
            print("cluster_size: ", cluster_size)

            if cluster_size == 1:
                points_to_remove_df = pd.DataFrame()
                points_to_remove_df["indices"] = [0]
                for eps in self.args.eps_list:
                    points_to_remove_df[f"eps={eps}"] = [False]
                if self.args.save_loc != "":
                    # Save dataframe
                    with open(df_file_loc, "wb") as file:
                        pickle.dump(points_to_remove_df, file)
                print("DONE cluster_id ", cluster_id)
                continue

            # By default, we keep hard examples from groups
            cluster_items_indices = list(range(cluster_size))

            # Shuffle cluster to keep random example from each group
            if self.args.which_to_keep.lower() == "random":
                random.shuffle(cluster_items_indices)
                cluster_i = cluster_i[cluster_items_indices]

            # Reverse cluster to keep easy examples
            if self.args.which_to_keep.lower() == "easy":
                cluster_items_indices = cluster_items_indices[::-1]
                cluster_i = cluster_i[cluster_items_indices]

            # Indices for cluster items in the dataset
            cluster_ids = cluster_i[:, 1].astype("int32")
            cluster_reps = embs[cluster_ids]
            cluster_reps = torch.tensor(cluster_reps)

            M = self.semdedup(cluster_i, cluster_reps, device)

            points_to_remove_df = pd.DataFrame()
            points_to_remove_df["indices"] = cluster_items_indices

            for eps in self.args.eps_list:
                # Remove a point from the dataset when its pairwise similarity to other points is > 1 - eps
                eps_points_to_remove = M > 1 - eps
                points_to_remove_df[f"eps={eps}"] = eps_points_to_remove

            if self.args.save_loc != "":
                # Save dataframe
                with open(df_file_loc, "wb") as file:
                    pickle.dump(points_to_remove_df, file)

            step_time.append(time.time() - step_st)
            print("DONE cluster: ", cluster_id)

        if len(step_time) > 0:
            avg_step_time = sum(step_time) / len(step_time)
        else:
            avg_step_time = 0

        print(
            f"DONE in {((time.time()-st)/60):.2f} minutes, Average Step time {avg_step_time:.2f}(s)"
        )
        return

    def __call__(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(self.args))
        job_start_cluster = self.job_start_cluster

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        print(
            f"This job will process clusters {job_start_cluster} to  {min(self.args.num_clusters, job_start_cluster + self.args.clusters_per_job)}"
        )

        num_tasks = world_size
        task_rank = rank

        print(f"There are {num_tasks} tasks in this job")
        print(f"I'm the task #{task_rank} on node {os.uname()[1]}")

        # Divide clusters among tasks
        num_clusters_per_task = int(math.ceil(self.args.clusters_per_job / num_tasks))
        start_cluster = job_start_cluster + task_rank * num_clusters_per_task
        end_cluster = job_start_cluster + (task_rank + 1) * num_clusters_per_task
        end_cluster = min(self.args.num_clusters, end_cluster)
        end_cluster = min(end_cluster, job_start_cluster + self.args.clusters_per_job)

        print(
            f"This task will process {num_clusters_per_task} clusters: cluster {start_cluster} to cluster {end_cluster}"
        )
        print(
            f"This task will process cluster {start_cluster} to cluster {end_cluster}"
        )

        self._process_shard(start_cluster, end_cluster)


def run_job(rank, world_size, args, job_start_cluster):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)

    job = SemDeDupJob(args, job_start_cluster)
    job()
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    confg_file = "semdedup_configs.yaml"
    with open(confg_file, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    for key, value in params.items():
        if isinstance(value, list):
            parser.add_argument(
                f"--{key}",
                nargs="+",
                type=type(value[0]),
                default=value,
            )
        else:
            parser.add_argument(
                f"--{key}",
                type=type(value),
                default=value,
            )
    args = parser.parse_args()
    print(args)

    os.makedirs(args.save_loc + "/dataframes", exist_ok=True)

    job_start_cluster = 0
    world_size = 8
    dist.init_process_group(backend="nccl")

    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    if torch.cuda.is_available():
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)

    job = SemDeDupJob(args, job_start_cluster)
    job()
