import numpy as np
import abc
from src.helper_subspace_functions import (
    multi_level_region_union_Nd,
    multi_level_region_intersection_Nd,
    convert_y_for_optimization,
    obtain_discrete_pareto_optima,
)


class SubsetAlgorithm(abc.ABC):
    def __init__(self, scalers):
        self.x_scaler = scalers[0]
        self.y_scaler = scalers[1]

    def unnormalize(self, x, y):
        return self.x_scaler.inverse_transform(x), self.y_scaler.inverse_transform(y)

    @abc.abstractmethod
    def identify_subspace(self, x, y):
        pass


class MultiRegionSetUnion(SubsetAlgorithm):
    def __init__(self, threshold_list, scalers):
        super().__init__(scalers)
        self.threshold_list = threshold_list

    def identify_subspace(self, x, y):
        x_unnorm, y_unnorm = super().unnormalize(x, y)
        desired_indices = multi_level_region_union_Nd(y_unnorm, self.threshold_list)

        return desired_indices


class MultiRegionSetIntersection(SubsetAlgorithm):
    def __init__(self, threshold_list, scalers):
        super().__init__(scalers)
        self.threshold_list = threshold_list

    def identify_subspace(self, x, y):
        x_unnorm, y_unnorm = super().unnormalize(x, y)
        desired_indices = multi_level_region_intersection_Nd(y_unnorm, self.threshold_list)
        return desired_indices


class Wishlist(SubsetAlgorithm):
    def __init__(self, threshold_bounds, scalers):
        super().__init__(scalers)
        self.threshold_bounds = threshold_bounds
        self.scalers = scalers

    def identify_subspace(self, x, y):
        desired_indices = set()
        for threshold_list in self.threshold_bounds:
            multi_region = MultiRegionSetIntersection(threshold_list, self.scalers)
            ids = multi_region.identify_subspace(x, y)
            desired_indices = desired_indices.union(set(ids))
        return list(desired_indices)


class GlobalOptimization1D(SubsetAlgorithm):
    def identify_subspace(self, x, y):
        desired_indices = [np.argmax(y)]
        return desired_indices


class ParetoFront(SubsetAlgorithm):
    def __init__(self, tolerance_list, max_or_min_list, scalers):
        super().__init__(scalers)
        self.tolerance_list = tolerance_list
        self.max_or_min_list = max_or_min_list

    def identify_subspace(self, x, y):
        x_unnorm, y_unnorm = super().unnormalize(x, y)
        y_unnorm = convert_y_for_optimization(y_unnorm, self.max_or_min_list)
        error_bars = self.tolerance_list * np.ones(np.array(y_unnorm).shape)
        desired_indices = obtain_discrete_pareto_optima(np.array(x_unnorm), np.array(y_unnorm), error_bars=error_bars)
        return np.array(desired_indices, dtype=int)
