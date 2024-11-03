import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, Matern, RBF
import pickle


class MGPR:
    """
    Class for multiple independent sklearn Gaussian Process Regression models.
    """

    def __init__(
        self, kernel_list: list, n_restarts_optimizer: int = 1
    ):  # TODO a list of what?
        self.models = [
            GaussianProcessRegressor(
                kernel=k, n_restarts_optimizer=n_restarts_optimizer
            )
            for k in kernel_list
        ]

    def fit(self, X: np.ndarray, y: np.ndarray):
        for i, model in enumerate(self.models):
            model.fit(X, y[:, i])

    # posterior mean for each measurable property
    def predict(self, X: np.ndarray):
        posterior_means = []
        posterior_stds = []
        for model in self.models:
            posterior_mean, posterior_std = model.predict(X, return_std=True)
            posterior_means.append(posterior_mean)
            posterior_stds.append(posterior_std)
        return np.array(posterior_means).T, np.array(posterior_stds).T

    # generate posterior samples for each measurable property
    def sample_y(self, x_domain: np.ndarray, n_samples: int) -> np.ndarray:
        posterior_samples_list = []

        for model in self.models:
            posterior_samples = model.sample_y(x_domain, n_samples)
            posterior_samples_list.append(posterior_samples)

        posterior_samples_array = np.moveaxis(np.array(posterior_samples_list), 1, 0)

        return posterior_samples_array

    def save(self, filename):
        print(f"Saving class instance to {filename}...")
        try:
            with open(filename, "wb") as f:
                pickle.dump(self, f)
            print("Class instance saved successfully!")
        except Exception as e:
            print(f"Failed to save class instance: {e}")

    @classmethod
    def load(cls, filename):
        print(f"Loading class instance from {filename}...")
        try:
            with open(filename, "rb") as f:
                instance = pickle.load(f)
            print("Class instance loaded successfully!")
            return instance
        except Exception as e:
            print(f"Failed to load class instance: {e}")
            return None


def fit_matern_hypers(
    x_train: np.ndarray,
    y_train: np.ndarray,
    kernel_list: list,  # TODO list of what?
    n_restarts_optimizer: int = 30,
) -> list:  # TODO list of what?
    """GP hyperparameter fitting for a kernel of the form ConstantKernel() * Matern() + WhiteNoise()

    Args:
        x_train (np.ndarray): current collected x data
        y_train (np.ndarray): current collected y data
        kernel_list (list): list of kernels for different measured properties

    Returns:
        list: list of kernels with fitted hyperparameters
    """
    multi_gpr = MGPR(kernel_list=kernel_list, n_restarts_optimizer=n_restarts_optimizer)
    multi_gpr.fit(x_train, y_train)  # update posterior distribution and fit GP hypers

    kernels = []
    for model in multi_gpr.models:
        params = model.kernel_.get_params()
        alpha = params["k1__k1__constant_value"]  # kernel variance
        ls = params["k1__k2__length_scale"]  # anisotropic lengthscales
        noise = params["k2__noise_level"]  # likelihood variance

        # this is neccesary to force all posterior samples to have the same GP hyperparameters
        k = ConstantKernel(
            constant_value=alpha, constant_value_bounds="fixed"
        ) * Matern(
            nu=5 / 2, length_scale=ls, length_scale_bounds="fixed"
        ) + WhiteKernel(
            noise, noise_level_bounds="fixed"
        )
        kernels.append(k)

    def save_model(self, fname):
        pass

    return kernels
