from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import ternary


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
    n_initial,
):
    fig, axes = plt.subplots(3, 3, dpi=100, figsize=(18, 18))
    X_unnorm = x_scaler.inverse_transform(X)

    if X_unnorm.shape[1] > 2:
        tsne = TSNE(n_components=2)
        X_unnorm = tsne.fit_transform(X_unnorm)  #

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
    try:
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
        axes[1, 1].scatter(
            Y_unnorm[collected_ids, 0][-1], Y_unnorm[collected_ids, 1][-1], color="purple", edgecolor="k"
        )
        axes[1, 1].set_xlabel("Sampling in measured property space")

        axes[1, 2].scatter(Y_unnorm[:, 0], Y_unnorm[:, 1], color="blue", alpha=0.1)
        axes[1, 2].scatter(
            Y_unnorm[true_target_ids, 0], Y_unnorm[true_target_ids, 1], color="orange", marker="D", edgecolor="k"
        )
        axes[1, 2].scatter(
            Y_unnorm[predicted_target_ids, 0], Y_unnorm[predicted_target_ids, 1], color="maroon", marker="x"
        )
        axes[1, 2].set_xlabel("Posterior mean predicted targets (measured property space)")
    except:
        pass

    # Plotting acquisition function and terms
    axes[2, 0].scatter(X_unnorm[:, 0], X_unnorm[:, 1], c=acquisition_function)
    axes[2, 0].set_xlabel("Acquisition Function")

    axes[2, 1].plot(np.arange(n_initial, n_initial + len(n_obtained_list)), n_obtained_list, color="k")
    axes[2, 1].plot(best_possible_n_obtained, color="r")
    axes[2, 1].plot(random_sampling, color="r")
    axes[2, 1].set_xlabel("Dataset Size")
    axes[2, 1].set_ylabel("Number Obtained")
    axes[2, 1].set_ylim(0, 1.1 * len(true_target_ids))
    axes[2, 1].set_xlim(0, len(collected_ids) + 5)

    axes[2, 2].axhline(y=1.0, color="r", linestyle="--")
    axes[2, 2].plot(np.arange(n_initial, n_initial + len(jaccard_posterior_list)), jaccard_posterior_list, color="k")
    axes[2, 2].set_xlabel("Dataset Size")
    axes[2, 2].set_ylabel("Jaccard Posterior Index")

    plt.show()


def plot_final_metrics(n_iters, metrics, strategies, best_possible_n_obtained, random_sampling):
    # Plot for "Number Obtained"
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Subplot for mean and std of "Number Obtained"
    for strategy in strategies:
        axes[0].plot(range(n_iters), np.mean(metrics[strategy]["n_obtained"], axis=0), label=strategy)
        axes[0].fill_between(
            range(n_iters),
            np.mean(metrics[strategy]["n_obtained"], axis=0) - np.std(metrics[strategy]["n_obtained"], axis=0),
            np.mean(metrics[strategy]["n_obtained"], axis=0) + np.std(metrics[strategy]["n_obtained"], axis=0),
            alpha=0.2,
        )

    axes[0].plot(best_possible_n_obtained, "r--", label="Best Possible")
    axes[0].plot(random_sampling, "r--", label="Random Sampling")
    axes[0].set_xlabel("Iteration Number")
    axes[0].set_ylabel("Number Obtained")
    axes[0].legend()

    # Subplot for "Jaccard Posterior Index"
    for strategy in strategies:
        axes[1].plot(range(n_iters), np.mean(metrics[strategy]["jaccard_posterior_index"], axis=0), label=strategy)
        axes[1].fill_between(
            range(n_iters),
            np.mean(metrics[strategy]["jaccard_posterior_index"], axis=0)
            - np.std(metrics[strategy]["jaccard_posterior_index"], axis=0),
            np.mean(metrics[strategy]["jaccard_posterior_index"], axis=0)
            + np.std(metrics[strategy]["jaccard_posterior_index"], axis=0),
            alpha=0.1,
        )

    axes[1].axhline(y=1.0, color="r", linestyle="-", label="Threshold")
    axes[1].set_xlabel("Iteration Number")
    axes[1].set_ylabel("Jaccard Posterior Index")
    axes[1].legend()

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Show the plots
    plt.show()


def plot_algo_true_function(algorithm, x_scaler, y_scaler, X, Y, posterior_mean=None, posterior_samples=None):
    target_subset_ids = algorithm.identify_subspace(f_x=Y, x=X)

    X_unnorm = x_scaler.inverse_transform(X)
    Y_unnorm = y_scaler.inverse_transform(Y)

    target_y = Y_unnorm[target_subset_ids]

    figure, ax = plt.subplots(figsize=(10 / 4, 10 / 4), dpi=200)
    fig, tax = ternary.figure(ax)
    x_all = np.hstack((X_unnorm, np.expand_dims(1 - np.sum(X_unnorm, axis=1), axis=-1)))

    tax.scatter(x_all / 100, c="#5F9AFF", s=8, alpha=0.5, linewidth=0.1, edgecolor="k", label="All points")
    tax.scatter(
        x_all[target_subset_ids] / 100,
        c="#FFD864",
        s=10,
        marker="D",
        linewidth=0.25,
        edgecolor="k",
        label="Desired points",
    )
    if posterior_mean is not None:
        posterior_mean_subset_ids = algorithm.identify_subspace(f_x=posterior_mean, x=X_unnorm)
        tax.scatter(
            x_all[posterior_mean_subset_ids] / 100,
            c="maroon",
            s=10,
            marker="x",
            linewidth=1.0,
            edgecolor="k",
            label="posterior mean points",
        )

    if posterior_samples is not None:
        for i in range(posterior_samples.shape[-1]):
            posterior_sample = posterior_samples[:, :, i]
            posterior_sample_subset_ids = algorithm.identify_subspace(f_x=posterior_sample, x=X_unnorm)
            tax.scatter(
                x_all[posterior_sample_subset_ids] / 100,
                s=10,
                marker="x",
                linewidth=1.0,
                edgecolor="k",
                label="posterior sample points",
            )

    tax.boundary(linewidth=1.0)
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")
    plt.tight_layout()
    tax.left_axis_label("Fe/Ni")
    tax.right_axis_label("Fe/Co")
    tax.bottom_axis_label("Ni/Co", position=(0.47, 0.05, 0))
    tax.show()

    if (posterior_samples is None) and (posterior_mean is None):
        fig = plt.figure(dpi=185)
        fig.set_figheight(10 / 4)
        fig.set_figwidth(10 / 4)
        plt.scatter(Y_unnorm[:, 0], Y_unnorm[:, 1], c="#5F9AFF", s=10, alpha=0.5, label="All design points")
        plt.scatter(
            target_y[:, 0],
            target_y[:, 1],
            c="#FFD864",
            s=10,
            linewidth=0.15,
            marker="D",
            edgecolor="k",
            label="Target design points",
        )

        plt.xlabel("Material Property 1")
        plt.ylabel("Material Property 2")
        plt.show()
