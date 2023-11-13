from matplotlib import pyplot as plt
import numpy as np


def plot_iteration_results(
    X,
    Y,
    x_scaler,
    y_scaler,
    collected_ids,
    true_target_ids,
    predicted_target_ids,
    acquisition_function,
    n_obtained_list,
    jaccard_posterior_list,
    best_possible_n_obtained,
    random_sampling,
):
    fig, axes = plt.subplots(4, 3, dpi=100, figsize=(18, 24))
    X_unnorm = x_scaler.inverse_transform(X)
    axes[0, 0].scatter(X_unnorm[:, 0], X_unnorm[:, 1], color="blue", alpha=0.3)
    axes[0, 0].scatter(
        X_unnorm[true_target_ids, 0], X_unnorm[true_target_ids, 1], color="orange", marker="D", edgecolor="k"
    )
    axes[0, 0].set_xlabel("Targets in design space")

    axes[0, 1].scatter(X_unnorm[:, 0], X_unnorm[:, 1], color="blue", alpha=0.3)
    axes[0, 1].scatter(
        X_unnorm[true_target_ids, 0], X_unnorm[true_target_ids, 1], color="orange", marker="D", edgecolor="k"
    )
    axes[0, 1].scatter(X_unnorm[collected_ids, 0], X_unnorm[collected_ids, 1], color="red", edgecolor="k")
    axes[0, 1].scatter(X_unnorm[collected_ids, 0][-1], X_unnorm[collected_ids, 1][-1], color="purple", edgecolor="k")
    axes[0, 1].set_xlabel("Collected data (design space)")

    axes[0, 2].scatter(X_unnorm[:, 0], X_unnorm[:, 1], color="blue", alpha=0.3)
    axes[0, 2].scatter(
        X_unnorm[true_target_ids, 0], X_unnorm[true_target_ids, 1], color="orange", marker="D", edgecolor="k"
    )
    axes[0, 2].scatter(
        X_unnorm[predicted_target_ids, 0], X_unnorm[predicted_target_ids, 1], color="maroon", marker="x"
    )
    axes[0, 2].set_xlabel("Posterior mean predicted targets (design space)")

    # Plotting in measured property space
    Y_unnorm = y_scaler.inverse_transform(Y)
    axes[1, 0].scatter(Y_unnorm[:, 0], Y_unnorm[:, 1], color="blue", alpha=0.1)
    axes[1, 0].scatter(
        Y_unnorm[true_target_ids, 0], Y_unnorm[true_target_ids, 1], color="orange", marker="D", edgecolor="k"
    )
    axes[1, 0].set_xlabel("Targets in measured property space")

    axes[1, 1].scatter(Y_unnorm[:, 0], Y_unnorm[:, 1], color="blue", alpha=0.1)
    axes[1, 1].scatter(
        Y_unnorm[true_target_ids, 0], Y_unnorm[true_target_ids, 1], color="orange", marker="D", edgecolor="k"
    )
    axes[1, 1].scatter(Y_unnorm[collected_ids, 0], Y_unnorm[collected_ids, 1], color="red", edgecolor="k")
    axes[1, 1].scatter(Y_unnorm[collected_ids, 0][-1], Y_unnorm[collected_ids, 1][-1], color="purple", edgecolor="k")
    axes[1, 1].set_xlabel("Sampling in measured property space")

    axes[1, 2].scatter(Y_unnorm[:, 0], Y_unnorm[:, 1], color="blue", alpha=0.1)
    axes[1, 2].scatter(
        Y_unnorm[true_target_ids, 0], Y_unnorm[true_target_ids, 1], color="orange", marker="D", edgecolor="k"
    )
    axes[1, 2].scatter(
        Y_unnorm[predicted_target_ids, 0], Y_unnorm[predicted_target_ids, 1], color="maroon", marker="x"
    )
    axes[1, 2].set_xlabel("Posterior mean predicted targets (measured property space)")

    # Plotting acquisition function and terms
    axes[2, 0].scatter(X_unnorm[:, 0], X_unnorm[:, 1], c=acquisition_function)
    axes[2, 0].set_xlabel("Acquisition Function")

    axes[3, 0].plot(np.arange(0, len(n_obtained_list)), n_obtained_list, color="k")
    axes[3, 0].plot(best_possible_n_obtained, color="r")
    axes[3, 0].plot(random_sampling, color="r")
    axes[3, 0].set_xlabel("Iteration Number")
    axes[3, 0].set_ylabel("Number Obtained")
    axes[3, 0].set_ylim(0, 1.1 * len(true_target_ids))
    axes[3, 0].set_xlim(0, len(collected_ids) + 5)

    axes[3, 1].axhline(y=1.0, color="r", linestyle="-")
    axes[3, 1].plot(np.arange(0, len(jaccard_posterior_list)), jaccard_posterior_list, color="k")
    axes[3, 1].set_xlabel("Iteration Number")
    axes[3, 1].set_ylabel("Jaccard Posterior Index")

    plt.show()


def plot_final_metrics(n_iters, metrics, strategies, best_possible_n_obtained, random_sampling):
    for strategy in strategies:
        plt.plot(range(n_iters), np.mean(metrics[strategy]["n_obtained"], axis=0), label=strategy)
        plt.fill_between(
            range(n_iters),
            np.mean(metrics[strategy]["n_obtained"], axis=0) - np.std(metrics[strategy]["n_obtained"], axis=0),
            np.mean(metrics[strategy]["n_obtained"], axis=0) + np.std(metrics[strategy]["n_obtained"], axis=0),
            alpha=0.2,
        )
    plt.plot(best_possible_n_obtained, "r--", label="Best Possible")
    plt.plot(random_sampling, "r--", label="Random Sampling")
    plt.xlabel("Iteration Number")
    plt.ylabel("Number Obtained")
    plt.legend()
    plt.show()

    for strategy in strategies:
        plt.plot(range(n_iters), np.mean(metrics[strategy]["jaccard_posterior_index"], axis=0), c="b", label="InfoBAX")
        plt.fill_between(
            range(n_iters),
            np.mean(metrics[strategy]["jaccard_posterior_index"], axis=0)
            - np.std(metrics[strategy]["jaccard_posterior_index"], axis=0),
            np.mean(metrics[strategy]["jaccard_posterior_index"], axis=0)
            + np.std(metrics[strategy]["jaccard_posterior_index"], axis=0),
            alpha=0.1,
            color="b",
        )
    plt.axhline(y=1.0, color="r", linestyle="-", label="Threshold")
    plt.xlabel("Iteration Number")
    plt.ylabel("Jaccard Posterior Index")
    plt.legend()
    plt.show()
