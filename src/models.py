import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor


class MGPR:
    def __init__(self, kernel_list):
        self.models = [
            GaussianProcessRegressor(kernel=kernel_list[i], n_restarts_optimizer=10, random_state=47)
            for i in range(len(kernel_list))
        ]

    def fit(self, X, y):
        for i, model in enumerate(self.models):
            model.fit(X, y[:, i])

    def predict(self, X, return_std=True):
        posterior_means = []
        posterior_stds = []
        for model in self.models:
            posterior_mean, posterior_std = model.predict(X, return_std=return_std)
            posterior_means.append(posterior_mean)
            posterior_stds.append(posterior_std)
        return np.array(posterior_means).T, np.array(posterior_stds).T

    def sample_y(self, x_domain, n_samples):
        posterior_samples_list = []

        for model in self.models:
            posterior_samples = model.sample_y(x_domain, n_samples)
            posterior_samples_list.append(posterior_samples)
        return np.moveaxis(np.array(posterior_samples_list), 1, 0)
