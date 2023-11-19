"""
This file contains helper functions to be used
to define classes in bax_multiproperty_algorithms.py
Mostly to seperate the individual algos and the internal 
logic
"""

from typing import List
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# make sure scaler for normalization is correcty passed to subset algo class
def assert_sklearn_scalers(scalers):
    for scaler in scalers:
        assert isinstance(
            scaler, (StandardScaler, MinMaxScaler)
        ), f"{type(scaler)} is not a valid scikit-learn scaler."


# Implementation adapted from: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python (Peter)
def obtain_discrete_pareto_optima(x=None, y=None, error_bars=None, use_errors=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(y.shape[0])
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(y):
        nondominated_point_mask = np.any(y >= y[next_point_index] - error_bars[next_point_index], axis=1)
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        y = y[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    else:
        return is_efficient


def level_region_1d(y_vals, thresholds):
    """Identify region of thresholds[0] < y_vals < thresholds[1]

    Args:
        y_vals (_type_): objective values
        thresholds (list/array): bounds of y

    Returns:
        list: indices of y satisfying thresholds[0] < y_vals < thresholds[1]
    """

    assert len(thresholds) == 2, "Need to have as many thresholds as dim_y"

    threshold_lower = thresholds[0]
    threshold_upper = thresholds[1]

    assert threshold_lower <= threshold_upper, "Upper bound >= lower bound"

    s1 = set((np.where(y_vals <= threshold_upper)[0]).tolist())
    s2 = set((np.where(y_vals >= threshold_lower)[0]).tolist())

    return list(s1.intersection(s2))


def multi_level_region_union_Nd(y_vals_array, thresholds_list):
    """Multi Dim generalization of level_region_1d with Union of levels

    Args:
        y_vals_array (NDArray): Objective function values
        thresholds_list (list[list[int]]): List of thresholds for all y_dims

    Returns:
        list: indices of y satisfying thresholds
    """
    y_vals_array = np.array(y_vals_array)
    thresholds_list = np.array(thresholds_list)

    if (y_vals_array.ndim == 1) and (np.array(thresholds_list).ndim == 1):
        return level_region_1d(y_vals_array, thresholds_list)

    assert y_vals_array.shape[1] == len(thresholds_list), "As many threshold lists as y_vals_array"

    n_dims_y = y_vals_array.shape[1]

    desired_points = set()

    for i in range(n_dims_y):
        y_vals = y_vals_array[:, i]
        thresholds = thresholds_list[i]
        desired_points = desired_points.union(set(level_region_1d(y_vals, thresholds)))

    return list(desired_points)


def multi_level_region_intersection_Nd(y_vals_array, thresholds_list):
    """Multi Dim generalization of level_region_1d with Intersection of levels

    Args:
        y_vals_array (NDArray): Objective function values
        thresholds_list (list[list[int]]): List of thresholds for all y_dims

    Returns:
        list: indices of y satisfying thresholds
    """
    y_vals_array = np.array(y_vals_array)
    thresholds_list = np.array(thresholds_list)

    if (y_vals_array.ndim == 1) and (np.array(thresholds_list).ndim == 1):
        return level_region_1d(y_vals_array, thresholds_list)

    assert y_vals_array.shape[1] == len(thresholds_list), "As many threshold lists as y_vals_array"

    n_dims_y = y_vals_array.shape[1]

    for i in range(n_dims_y):
        y_vals = y_vals_array[:, i]
        thresholds = thresholds_list[i]

        if i == 0:
            desired_points = set(level_region_1d(y_vals, thresholds))
        else:
            desired_points = desired_points.intersection(set(level_region_1d(y_vals, thresholds)))

    return list(desired_points)


def sublevelset(y_vals, threshold):
    """y_vals < threshold

    Args:
        y_vals (_type_): Objective values
        threshold (_type_): Threshold

    Returns:
        list: indices of y satisfying thresholds
    """

    s1 = set((np.where(y_vals <= threshold)[0]).tolist())
    s2 = set((np.where(y_vals >= 0)[0]).tolist())

    return list(s1.intersection(s2))


def discontinous_library_1d(y_vals, list_of_vals, eps_vals: List[float] = None):
    """Identifies a Discontinous Library (Set of vals + epsilon) of an array in y

    Args:
        y_vals (Array): _description_
        list_of_vals (List): List of Desired Values
        eps_vals (List[float], optional): Tolerance for each value. Defaults to None.

    Returns:
        list: indices of y satisfying thresholds
    """
    y_vals = np.array(y_vals)
    list_of_vals = np.array(list_of_vals)

    if eps_vals is None:
        eps_vals = np.ones(list_of_vals.shape) * 0.1
    elif isinstance(eps_vals, (float, int)):
        eps_vals = np.ones(list_of_vals.shape) * eps_vals
    else:
        eps_vals = np.array(eps_vals)

    assert len(list_of_vals) == len(eps_vals), "eps_vals must be same size as list_of_vals"

    point_set = set()

    for ct, val in enumerate(list_of_vals):
        id1 = set(list(np.where(y_vals <= val + eps_vals[ct])[0]))
        id2 = set(list(np.where(y_vals >= val - eps_vals[ct])[0]))

        z = list(id1.intersection(id2))
        if len(z) > 0:
            for elem in z:
                point_set.add(elem)
    return list(point_set)


def convert_y_for_optimization(y, max_or_min_list):
    """Function which negates columns of y depending on whether the user wants to minimize or maximize a property"""
    y = np.array(y)
    max_or_min_list = np.array(max_or_min_list)

    assert np.array_equal(max_or_min_list, max_or_min_list.astype(bool)), "The array is not boolean."
    assert y.shape[1] == len(max_or_min_list), "Please note whether each property should be maximized or minimized"

    for i, max_min in enumerate(max_or_min_list):
        if max_min == 0:
            y[:, i] = -y[:, i]

    return y
