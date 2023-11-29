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
        self.user_algo_params = user_algo_params  # parameters for user-specific algorithm
        self.scalers = self.user_algo_params["scalers"]  # x, y sklearn standard scalers
        assert_sklearn_scalers(self.scalers)

    # allow user to specify bound thresholds in unnormalized space
    def unnormalize(self, x, f_x):
        y_scaler = self.scalers[1]
        x_scaler = self.scalers[0]
        return x_scaler.inverse_transform(x), y_scaler.inverse_transform(f_x)

    # executes the user-algorithm
    def identify_subspace(self, f_x, x):
        x_unnorm, f_x_unnorm = self.unnormalize(x, f_x)
        list_of_target_indices = self.user_algorithm(f_x_unnorm, x_unnorm)
        return list(set(list_of_target_indices))  # ensure that the return is unique

    @abc.abstractmethod
    def user_algorithm(self, f_x, x):
        pass


class MultibandUnion(SubsetAlgorithm):
    """
    Union of multiple level bands for each property
    """

    def __init__(self, user_algo_params):
        super().__init__(user_algo_params)

    def user_algorithm(self, f_x, x):
        threshold_bands = self.user_algo_params["threshold_bands"]
        list_of_target_indices = multi_level_region_union_Nd(f_x, threshold_bands)
        return list_of_target_indices


class MultibandIntersection(SubsetAlgorithm):
    """
    Intersection of multiple level bands for each property
    """

    def __init__(self, user_algo_params):
        super().__init__(user_algo_params)

    def user_algorithm(self, f_x, x):
        threshold_bounds = self.user_algo_params["threshold_bounds"]
        list_of_target_indices = multi_level_region_intersection_Nd(f_x, threshold_bounds)
        return list_of_target_indices


class ConditionalMultiband(SubsetAlgorithm):
    def __init__(self, user_algo_params):
        super().__init__(user_algo_params)

    def user_algorithm(self, f_x, x):
        threshold_bounds = self.user_algo_params["threshold_bounds"]
        list_of_target_indices = multi_level_region_intersection_Nd(f_x, threshold_bounds)

        # return a sublevel set in one property if the intersection is empty
        if len(list_of_target_indices) == 0:
            p2_min, p2_max = threshold_bounds[1]
            list_of_target_indices = sublevelset(f_x, p2_max)

        return list_of_target_indices


class Wishlist(SubsetAlgorithm):
    """
    Composition of multiple multibands
    """

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
    """
    Global optimization algorithm. This is similar to Entropy Search (ES) technique.
    """

    def __init__(self, user_algo_params):
        super().__init__(user_algo_params)

    def user_algorithm(self, f_x, x):
        assert f_x.shape[1] == 1.0
        maximize_fn = self.user_algo_params["maximize_fn"]
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
        return list(desired_indices)


class ParetoPlusMultiband(SubsetAlgorithm):
    def __init__(self, user_algo_params):
        super().__init__(user_algo_params)

    def user_algorithm(self, f_x, x):
        max_or_min_list = self.user_algo_params["max_or_min_list"]
        tolerance_list = self.user_algo_params["tolerance_list"]
        threshold_bounds = self.user_algo_params["threshold_bounds"]

        f_x_converted = convert_y_for_optimization(f_x, max_or_min_list)
        error_bars = tolerance_list * np.ones(np.array(f_x_converted).shape)

        t1 = obtain_discrete_pareto_optima(np.array(x), np.array(f_x_converted), error_bars=error_bars)
        t2 = multi_level_region_intersection_Nd(f_x, threshold_bounds)

        return list(set(t1).union(set(t2)))


class PercentileSet1D(SubsetAlgorithm):
    """
    Algorithm which finds the top k% of points in a design space
    """

    def __init__(self, user_algo_params):
        super().__init__(user_algo_params)

    def user_algorithm(self, f_x, x):
        assert f_x.shape[1] == 1.0
        percentile_threshold = self.user_algo_params["percentile_threshold"]
        top_percentile_value = np.percentile(f_x, percentile_threshold)
        target_ids = list(set(np.where(f_x >= top_percentile_value)[0]))
        return target_ids


class MonodisperseLibrary(SubsetAlgorithm):
    """
    Custom algorithm to find monodisperse nanoparticles with a variety of radii
    """

    def __init__(self, user_algo_params):
        super().__init__(user_algo_params)

    def user_algorithm(self, f_x, x):
        pd_threshold = self.user_algo_params["polysdispersity_threshold"]
        radii_list = self.user_algo_params["target_radii_list"]
        radii_tol = self.user_algo_params["target_radii_tol"]

        y1 = np.array(f_x)[:, 0]
        y2 = np.array(f_x)[:, 1]

        # intersection of level set and disconnected list
        intersect_id = set(sublevelset(y2, pd_threshold)).intersection(
            set(discontinous_library_1d(y1, radii_list, eps_vals=radii_tol))
        )

        return list(intersect_id)
