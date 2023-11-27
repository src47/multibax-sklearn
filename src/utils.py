from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from scipy.io import loadmat


def XY_from_csv(path_to_csv: str, columns_x: List[str], columns_y: List[str]):
    # Load the data from the CSV file
    df = pd.read_csv(path_to_csv)

    # assert that there are no NaNs in the dataframe (chatgpt)
    assert df.isna().sum().sum() == 0, "Should not have NANs in the dataset"

    # assert that df does not have any repeated rows (chatgpt)
    assert df.duplicated().sum() == 0, "Dataset should not have repeated rows"

    X = np.array(df[columns_x])
    y = np.array(df[columns_y])

    assert X.shape[1] >= 2, "Input X must be at least two dimensional"
    return X, y


def random_sampling_no_replace(n_iterations: int, n_targets: int, iteration: int) -> float:
    rv = hypergeom(M=n_iterations, n=n_targets, N=iteration)
    x = np.arange(0, n_targets + 1)  # all possible number of desired points obtained
    pmf = rv.pmf(x)
    mean = np.sum(x * pmf)  # average number of desired points obtained
    return mean


# load ternary data file
def load_ternary_data(file):
    data = loadmat(file)
    X = data["C"][:, 0:2]  # only need 2 dimensions of composition since c3 = c2 + c1
    Y1 = data["Coer"]  # magnetic property 1
    Y2 = data["Kerr"]  # magnetic property 2
    Y = np.hstack((Y1, Y2))
    return X, Y
