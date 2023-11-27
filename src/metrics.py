"""
Metrics to characterize performance of data acquisition
"""


# Number of measured points that meet the experimental goal
def get_n_obtained(collected_ids: list, true_target_ids: list) -> int:
    n_obtained = len(set(collected_ids).intersection(set(true_target_ids)))
    return n_obtained


# Intersection / Union of ground-truth targets and predicted targets based on posterior mean
def get_jaccard_posterior(predicted_target_ids: list, true_target_ids: list) -> float:
    jaccard = len(set(predicted_target_ids).intersection(set(true_target_ids))) / len(
        set(predicted_target_ids).union(set(true_target_ids))
    )
    return jaccard
