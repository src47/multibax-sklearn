from copy import deepcopy
import numpy as np
from tqdm import tqdm


def get_posterior_mean_and_std(x, model):
    posterior_mean, posterior_std = np.squeeze(model.predict(x, return_std=True))
    return posterior_mean, posterior_std


def calculate_entropy(x, model):
    posterior_mean, posterior_std = get_posterior_mean_and_std(x, model)
    entropy = 0.5 * np.log(2 * np.pi * (posterior_std**2)) + 0.5
    return entropy


def multiproperty_infobax(x_domain, x_train, y_train, model, algorithm, n_posterior_samples=20, verbose=False):
    multi_gpr_model = deepcopy(model)
    multi_gpr_model.fit(x_train, y_train)

    # essentially the same as US
    term1 = calculate_entropy(x_domain, multi_gpr_model)

    term2 = np.zeros(term1.shape)
    posterior_samples = multi_gpr_model.sample_y(x_domain, n_posterior_samples)
    if len(posterior_samples.shape) == 2:
        posterior_samples = np.expand_dims(posterior_samples, axis=1)

    for i in tqdm(range(n_posterior_samples)) if verbose else range(n_posterior_samples):
        multi_gpr_model_fake = deepcopy(model)
        posterior_sample = posterior_samples[:, :, i]
        predicted_target_ids = algorithm.identify_subspace(x=x_domain, y=posterior_sample)

        # add in "hallucinated data" based on algorithm execution on posterior sample
        if len(predicted_target_ids) != 0:
            desired_x = x_domain[predicted_target_ids]
            predicted_desired_y = posterior_sample[predicted_target_ids]
            fake_x_train = np.vstack((x_train, desired_x))
            fake_y_train = np.vstack((y_train, predicted_desired_y))
        else:
            fake_x_train = x_train
            fake_y_train = y_train

        multi_gpr_model_fake.fit(fake_x_train, fake_y_train)

        term2 += calculate_entropy(x_domain, multi_gpr_model_fake)

    # takes into account experimental goal
    term2 = -(1 / n_posterior_samples) * term2

    acquisition_values = term1 + term2

    if len(acquisition_values.shape) > 1:
        acquisition_values = np.mean(acquisition_values, axis=-1)
        term1 = np.mean(term1, axis=-1)
        term2 = np.mean(term2, axis=-1)

    return acquisition_values, multi_gpr_model, term1, term2


def mean_posterior_std(posterior_std):
    return np.mean(posterior_std, axis=-1)


def meanbax_stuck(predicted_target_ids, collected_ids):
    return (set(predicted_target_ids).issubset(collected_ids)) or (len(set(predicted_target_ids)) == 0)


def multiproperty_meanbax(x_domain, x_train, y_train, model, algorithm, collected_ids):
    model.fit(x_train, y_train)
    posterior_mean, posterior_std = model.predict(x_domain)
    predicted_target_ids = algorithm.identify_subspace(x=x_domain, y=posterior_mean)

    if meanbax_stuck(predicted_target_ids, collected_ids):
        acquisition_function = mean_posterior_std
    else:
        acquisition_function = np.zeros(x_domain.shape[0])
        acquisition_function[predicted_target_ids] = mean_posterior_std(posterior_std)[predicted_target_ids]

    return acquisition_function, model


def mixed(x_domain, x_train, y_train, model, algorithm, n_posterior_samples, collected_ids):
    model.fit(x_train, y_train)
    posterior_mean, posterior_std = model.predict(x_domain)
    predicted_target_ids = algorithm.identify_subspace(x=x_domain, y=posterior_mean)

    if meanbax_stuck(predicted_target_ids, collected_ids):
        acquisition_function, model, term1, term2 = multiproperty_infobax(
            x_domain, x_train, y_train, model, algorithm, n_posterior_samples, verbose=False
        )
    else:
        acquisition_function = np.zeros(x_domain.shape[0])
        acquisition_function[predicted_target_ids] = mean_posterior_std(posterior_std)[predicted_target_ids]

    return acquisition_function, model


def run_acquisition(
    x_train, y_train, X, Y, strategy, algorithm, model, collected_ids, n_posterior_samples, prevent_requery=True
):
    if strategy == "infobax":
        acquisition_function, trained_model, term1, term2 = multiproperty_infobax(
            x_domain=X,
            x_train=x_train,
            y_train=y_train,
            model=model,
            algorithm=algorithm,
            n_posterior_samples=n_posterior_samples,
            verbose=False,
        )
    elif strategy == "meanbax":
        acquisition_function, trained_model = multiproperty_meanbax(
            x_domain=X, x_train=x_train, y_train=y_train, model=model, algorithm=algorithm, collected_ids=collected_ids
        )
    elif strategy == "mixed":
        acquisition_function, trained_model = mixed(
            x_domain=X,
            x_train=x_train,
            y_train=y_train,
            model=model,
            algorithm=algorithm,
            n_posterior_samples=n_posterior_samples,
            collected_ids=collected_ids,
        )
    else:
        raise Exception("Unknown acquisition function")

    next_id = optimize_acquisition_function(
        acquisition_function=acquisition_function, collected_ids=collected_ids, prevent_requery=prevent_requery
    )
    collected_ids.append(next_id)

    x_next = X[next_id]
    y_next = Y[next_id]

    x_train = np.vstack((x_train, x_next))
    y_train = np.vstack((y_train, y_next))

    return x_train, y_train, trained_model, collected_ids, acquisition_function


def optimize_acquisition_function(acquisition_function, collected_ids=None, prevent_requery=True):
    if (prevent_requery) and (collected_ids is not None):
        acquisition_function[collected_ids] = -np.inf

    next_id = np.argmax(acquisition_function)
    return next_id
