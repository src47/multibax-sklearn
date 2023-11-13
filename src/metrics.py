def get_n_obtained(collected_ids, true_target_ids):
    n_obtained = len(set(collected_ids).intersection(set(true_target_ids)))
    return n_obtained


def get_jaccard_posterior(predicted_target_ids, true_target_ids):
    jaccard = len(set(predicted_target_ids).intersection(set(true_target_ids))) / len(
        set(predicted_target_ids).union(set(true_target_ids))
    )
    return jaccard
