import numpy as np
import abc

from src.helper_subspace_functions import (
    assert_sklearn_scalers,
    multi_level_region_union_Nd,
    multi_level_region_intersection_Nd,
    convert_y_for_optimization,
    obtain_discrete_pareto_optima,
    sublevelset,
    discontinous_library_1d,
)


class SubsetAlgorithm(abc.ABC):
    def __init__(self, user_algo_params):
        self.user_algo_params = user_algo_params
        self.scalers = self.user_algo_params["scalers"]
        assert_sklearn_scalers(self.scalers)

    def unnormalize(self, x, f_x):
        y_scaler = self.scalers[1]
        x_scaler = self.scalers[0]
        return x_scaler.inverse_transform(x), y_scaler.inverse_transform(f_x)

    def identify_subspace(self, f_x, x):
        x_unnorm, f_x_unnorm = self.unnormalize(x, f_x)
        list_of_target_indices = self.user_algorithm(f_x_unnorm, x_unnorm)
        return list_of_target_indices

    @abc.abstractmethod
    def user_algorithm(self, f_x, x):
        pass


class MultibandUnion(SubsetAlgorithm):
    def __init__(self, user_algo_params):
        super().__init__(user_algo_params)

    def user_algorithm(self, f_x, x):
        threshold_bands = self.user_algo_params["threshold_bands"]
        list_of_target_indices = multi_level_region_union_Nd(f_x, threshold_bands)
        return list_of_target_indices


class MultibandIntersection(SubsetAlgorithm):
    def __init__(self, user_algo_params):
        super().__init__(user_algo_params)

    def user_algorithm(self, f_x, x):
        threshold_bands = self.user_algo_params["threshold_bands"]
        list_of_target_indices = multi_level_region_intersection_Nd(f_x, threshold_bands)
        return list_of_target_indices


class Wishlist(SubsetAlgorithm):
    def __init__(self, user_algo_params):
        super().__init__(user_algo_params)

    def user_algorithm(self, f_x, x):
        threshold_bounds_list = self.user_algo_params["threshold_bounds_list"]
        target_ids = set()

        for threshold_list in threshold_bounds_list:
            ids = multi_level_region_intersection_Nd(f_x, threshold_list)
            target_ids = target_ids.union(set(ids))

        return list(target_ids)


class GlobalOptimization1D(SubsetAlgorithm):
    def __init__(self, user_algo_params):
        super().__init__(user_algo_params)

    def user_algorithm(self, f_x, x):
        maximize_fn = self.user_algo_params("maximize_fn")
        if maximize_fn:
            max_value = np.max(f_x)
        elif maximize_fn is False:
            max_value = np.max(-f_x)
        else:
            raise Exception("maximize_fn must be of type bool")
        argmax_indices = np.where(f_x == max_value)[0]  # Get the indices where the values are equal to the maximum
        return argmax_indices


class ParetoFront(SubsetAlgorithm):
    def __init__(self, user_algo_params):
        super().__init__(user_algo_params)

    def user_algorithm(self, f_x, x):
        max_or_min_list = self.user_algo_params["max_or_min_list"]
        tolerance_list = self.user_algo_params["tolerance_list"]
        f_x_converted = convert_y_for_optimization(f_x, max_or_min_list)
        error_bars = tolerance_list * np.ones(np.array(f_x_converted).shape)
        desired_indices = obtain_discrete_pareto_optima(np.array(x), np.array(f_x_converted), error_bars=error_bars)
        return list(desired_indices, dtype=int)


class PercentileSet1D(SubsetAlgorithm):
    def __init__(self, user_algo_params):
        super().__init__(user_algo_params)

    def user_algorithm(self, f_x, x):
        percentile_threshold = self.user_algo_params["percentile_threshold"]
        top_percentile_value = np.percentile(f_x, percentile_threshold)
        target_ids = list(set(np.where(f_x >= top_percentile_value)[0]))
        return target_ids


class MonodisperseLibrary(SubsetAlgorithm):
    def __init__(self, user_algo_params):
        super().__init__(user_algo_params)

    def user_algorithm(self, f_x, x):
        polysdispersity_threshold = self.user_algo_params["polysdispersity_threshold"]
        target_radii_list = self.user_algo_params["target_radii_list"]
        target_radii_tol = self.user_algo_params["target_radii_tol"]

        y1 = np.array(f_x)[:, 0]
        y2 = np.array(f_x)[:, 1]

        # intersection of level set and disconnected list
        intersect_id = list(
            set(sublevelset(y2, polysdispersity_threshold)).intersection(
                set(discontinous_library_1d(y1, target_radii_list, eps_vals=target_radii_tol))
            )
        )

        return intersect_id
