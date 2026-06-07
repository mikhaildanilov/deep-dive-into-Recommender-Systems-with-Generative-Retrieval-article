"""Retrieval baselines for the Amazon Reviews dataset.

Each baseline is runnable as a standalone script and reports Recall@k / NDCG@k
on the validation and test splits using the repository's leave-one-out protocol
(test target = last item, validation target = second-to-last item).
"""
