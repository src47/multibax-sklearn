import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, Matern, RBF


class MGPR:
    def __init__(self, kernel_list: list, n_restarts_optimizer: int = 1):  # TODO a list of what?
        self.models = [
            GaussianProcessRegressor(kernel=kernel_list[i], n_restarts_optimizer=n_restarts_optimizer)
            for i in range(len(kernel_list))
        ]

    def fit(self, X: np.ndarray, y: np.ndarray):
        for i, model in enumerate(self.models):
            model.fit(X, y[:, i])

    def predict(self, X: np.ndarray):
        posterior_means = []
        posterior_stds = []
        for model in self.models:
            posterior_mean, posterior_std = model.predict(X, return_std=True)
            posterior_means.append(posterior_mean)
            posterior_stds.append(posterior_std)
        return np.array(posterior_means).T, np.array(posterior_stds).T

    def sample_y(self, x_domain: np.ndarray, n_samples: int) -> np.ndarray:
        posterior_samples_list = []

        for model in self.models:
            posterior_samples = model.sample_y(x_domain, n_samples)
            posterior_samples_list.append(posterior_samples)
        return np.moveaxis(np.array(posterior_samples_list), 1, 0)


def fit_hypers(
    x_train: np.ndarray,
    y_train: np.ndarray,
    kernel_list: list,  # TODO list of what?
    n_restarts_optimizer: int = 30,
) -> list:  # TODO list of what?
    multi_gpr = MGPR(kernel_list=kernel_list, n_restarts_optimizer=n_restarts_optimizer)
    multi_gpr.fit(x_train, y_train)

    kernels = []
    for i in range(len(multi_gpr.models)):
        params = multi_gpr.models[i].kernel_.get_params()
        alpha = params["k1__k1__constant_value"]
        ls = params["k1__k2__length_scale"]
        noise = params["k2__noise_level"]

        k = ConstantKernel(constant_value=alpha, constant_value_bounds="fixed") * Matern(
            nu=5 / 2, length_scale=ls, length_scale_bounds="fixed"
        ) + WhiteKernel(noise, noise_level_bounds="fixed")
        kernels.append(k)

    return kernels
