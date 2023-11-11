import numpy as np
from tqdm import tqdm
from copy import deepcopy


def get_posterior_mean_and_std(x, model):
    posterior_mean, posterior_std = np.squeeze(model.predict(x, return_std=True))
    return posterior_mean, posterior_std


def calculate_entropy(x, model):
    posterior_mean, posterior_std = get_posterior_mean_and_std(x, model)
    entropy = 0.5 * np.log(2 * np.pi * (posterior_std**2)) + 0.5
    return entropy


def multi_infobax_gp(x_domain, x_train, y_train, model, algorithm, n_posterior_samples=20, verbose=False):
    multi_gpr_model = deepcopy(model)
    multi_gpr_model.fit(x_train, y_train)

    term1 = calculate_entropy(x_domain, multi_gpr_model)
    term2 = np.zeros(term1.shape)

    posterior_samples = multi_gpr_model.sample_y(x_domain, n_posterior_samples)

    iteration = tqdm(range(n_posterior_samples)) if verbose else range(n_posterior_samples)

    for i in iteration:
        multi_gpr_model_fake = deepcopy(model)
        posterior_sample = posterior_samples[:, :, i]
        desired_indices = algorithm.identify_subspace(x=x_domain, y=posterior_sample)

        if len(desired_indices) != 0:
            desired_x = x_domain[desired_indices]
            predicted_desired_y = posterior_sample[desired_indices]
            fake_x_train = np.vstack((x_train, desired_x))
            fake_y_train = np.vstack((y_train, predicted_desired_y))
        else:
            fake_x_train = x_train
            fake_y_train = y_train

        multi_gpr_model_fake.fit(fake_x_train, fake_y_train)

        term2 += calculate_entropy(x_domain, multi_gpr_model_fake)

    term2 = -(1 / n_posterior_samples) * term2

    acquisition_values = term1 + term2

    if len(acquisition_values.shape) > 1:
        acquisition_values = np.mean(acquisition_values, axis=-1)
        term1 = np.mean(term1, axis=-1)
        term2 = np.mean(term2, axis=-1)

    return acquisition_values, multi_gpr_model, term1, term2
