from difflib import ndiff
from time import perf_counter

import datasets
import faiss
import numpy as np
from datasets import Dataset, load_dataset
from datasketch import MinHash, MinHashLSH
from model2vec import StaticModel
from reach import Reach
from tqdm import tqdm
from wordllama import WordLlama

model = StaticModel.from_pretrained("minishlab/M2V_base_output")
# ds = load_dataset("ag_news")["train"]
# texts = ds["text"]
# print(f"Number of docs: {len(texts)}")

# dataset = datasets.concatenate_datasets(
#     [
#         datasets.load_from_disk(
#             f"/data/users/zichunyu/data/fineweb/sample-350BT/train/0_embedding/{i}"
#         )
#         for i in range(1)
#     ]
# )
dataset_path = "/data/users/zichunyu/data/fineweb/sample-350BT/val_embedding.jsonl"
dataset = Dataset.from_json(dataset_path)
# print(np.array(dataset["__embedding"]).shape)
# embedding_matrix = model.encode(texts, show_progressbar=True)
embedding_matrix = np.array(dataset["__embedding"])
embedding_matrix = embedding_matrix[:50000]
print(f"Embedding matrix shape: {embedding_matrix.shape}")

ncentroids = 1
niter = 20
verbose = True
n = embedding_matrix.shape[0]
d = embedding_matrix.shape[1]


def deduplicate(
    embedding_matrix: np.ndarray, threshold: float, batch_size: int = 1024
) -> tuple[np.ndarray, dict[int, int]]:
    """
    Deduplicate embeddings and return the deduplicated indices and a mapping of removed indices to their corresponding original indices.

    :param embedding_matrix: The embeddings to deduplicate.
    :param threshold: The similarity threshold to use for deduplication.
    :param batch_size: The batch size to use for similarity computation.
    :return: A tuple containing the deduplicated indices and a dictionary mapping removed indices to original indices.
    """
    # reach = Reach(
    #     vectors=embedding_matrix, items=[str(i) for i in range(len(embedding_matrix))]
    # )

    # Use a set for deduplicated indices and keep track of duplicates
    deduplicated_indices = set(
        range(len(embedding_matrix))
    )  # Start with all indices as deduplicated
    duplicate_to_original_mapping = {}

    # results = reach.nearest_neighbor_threshold(
    #     embedding_matrix,
    #     threshold=threshold,
    #     batch_size=batch_size,
    #     show_progressbar=True,
    # )

    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
    index = faiss.IndexFlatL2(d)
    index.add(x)
    D, I = index.search(kmeans.centroids, 15)
    print(D.shape, I[:10])

    # Process duplicates
    for i, similar_items in enumerate(tqdm(results)):
        if i not in deduplicated_indices:
            continue  # Skip already marked duplicates

        # Similar items are returned as (index, score), we are only interested in the index
        similar_indices = [int(item[0]) for item in similar_items if int(item[0]) != i]

        # Mark similar documents as duplicates and map them to the original
        for sim_idx in similar_indices:
            if sim_idx in deduplicated_indices:
                deduplicated_indices.remove(sim_idx)
                duplicate_to_original_mapping[sim_idx] = i  # Map duplicate to original

    return np.array(list(deduplicated_indices)), duplicate_to_original_mapping


# 0.6 -> 1/8
deduplicated_indices, duplicate_to_original_mapping = deduplicate(
    embedding_matrix, threshold=0.6
)
# print(deduplicated_indices)
print(f"Number of deduplicated docs: {len(deduplicated_indices)}")
