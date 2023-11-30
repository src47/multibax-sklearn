[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10233974.svg)](https://doi.org/10.5281/zenodo.10233974)

# Targeted materials discovery using Bayesian Algorithm Execution

This repository implements various Bayesian Algorithm Execution acquisition strategies for precise, targeted chemical and materials discovery. Users state their experimental goal via a simple algorithm which is then automatically converted into a goal-aligned, intelligent data acquisition strategy. This framework applies to multi-property measurements and goes beyond the capabilities of multi-objective bayesian optimization.

<img width="1103" alt="Screen Shot 2023-11-17 at 8 11 07 PM" src="https://github.com/src47/sklearn-bax/assets/39596225/beeb53e1-bbe6-47c8-89a4-fefb510143a9">

## Installation

1) Make a new local folder and clone the repository

```
git clone https://github.com/src47/multibax-sklearn.git
```

2) Create a virtual environment and install requirements

```
python3 -m venv .venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Directories

**notebooks/tutorials** 

We highly recommend reviewing the following tutorial notebooks before using BAX acquisition functions for specific experimental goals. Note, [tutorial 4](notebooks/tutorials/tutorial_4_data_acquisition_using_BAX.ipynb) contains a full pipeline of data acquisition using BAX. 

1) [tutorial 1](notebooks/tutorials/tutorial_1_expressing_a_goal_as_an_algorithm.ipynb): This notebook explains how to pose an experimental goal as an algorithm and showcases running an algorithm on the ground-truth function.

2) [tutorial 2](notebooks/tutorials/tutorial_2_algorithm_execution_surrogate_model.ipynb): This notebook shows how to execute an algorithm on posterior draws from a trained surrogate Gaussian Process model. 

3) [tutorial 3](notebooks/tutorials/tutorial_3_defining_metrics.ipynb): This notebook explains two metrics (Number Obtained and Posterior Jaccard Index) used to assess the quality of data acquisition. 

4) [tutorial 4](notebooks/tutorials/tutorial_4_data_acquisition_using_BAX.ipynb): This notebook demonstrates using InfoBAX, MeanBAX and SwitchBAX to find a "wishlist" of regions for a magnetic materials characterization dataset. 


**src** 

- *acquisition.py*: Implementation of InfoBAX, MeanBAX, SwitchBAX and US. 
- *algorithms.py, helper_subspace_functions.py*: User algorithms for materials and chemical discovery. 
- *metrics.py*: Implementation of the Number Obtained and Posterior Jaccard Index metrics. 

## Citation

Please cite "_Targeted materials discovery using Bayesian algorithm execution_" (arxiv link to come!) if this repository was useful to your research.

## References

Methodology in this repo builds on InfoBAX [1] and Multi-point BAX [2]

[1] Neiswanger, Willie, et al. "Bayesian algorithm execution: Estimating computable properties of black-box functions using mutual information." International Conference on Machine Learning. PMLR, 2021.

[2] Miskovich, Sara A., et al. "Bayesian algorithm execution for tuning particle accelerator emittance with partial measurements." arXiv preprint arXiv:2209.04587 (2022).

**Please direct any questions or comments to chitturi@stanford.edu, akashr@stanford.edu or willie.neiswanger@gmail.com. 
