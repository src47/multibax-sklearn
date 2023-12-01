[![DOI](https://zenodo.org/badge/713948168.svg)](https://zenodo.org/doi/10.5281/zenodo.10233973)

# Multi-property Bayesian algorithm execution using sklearn

This repository implements various Bayesian algorithm execution (BAX) acquisition strategies for precise, targeted chemical and materials discovery. It enables a user to quickly isolate portions of a design space that meet highly customized goals. Scientific applications include finding the _set of all_ compounds which have measured properties that fall within a band of values (level band) or which fall in the top k percentile of a dataset (percentile band), identifying synthesis conditions that produce monodisperse nanoparticles with multiple precisely specified particle sizes, and finding chemically diverse sets of ligands that are strong, non-toxic binders. This framework applies to multi-property measurements and goes beyond the capabilities of multi-objective bayesian optimization. 

## Description

:writing_hand: Users state their experimental goal via a simple filtering algorithm which is able to return the correct subset of the design space *if the true underlying mapping were known* (**A**). 

<img width="1355" alt="Screen Shot 2023-12-01 at 6 57 37 AM" src="https://github.com/src47/multibax-sklearn/assets/39596225/38d7c0b6-a1ec-47b6-b299-7b396e999d75">


:mechanical_arm: The Bayesian algorithm execution procedure circumvents needing to actually know the true underlying mapping and automatically creates a goal-aligned, data acquisition strategy (**B-D**). 

<img width="1356" alt="Screen Shot 2023-12-01 at 6 57 55 AM" src="https://github.com/src47/multibax-sklearn/assets/39596225/88f9cdcd-144b-4f05-8e4d-bbddf604e7cd">

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

We highly recommend reviewing the following tutorial notebooks before using BAX acquisition functions for specific experimental goals. Note, [Tutorial 4](notebooks/tutorials/tutorial_4_data_acquisition_using_BAX.ipynb) contains a full pipeline of data acquisition using BAX. 

1) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/src47/multibax-sklearn/blob/main/notebooks/tutorials/tutorial_1_expressing_a_goal_as_an_algorithm.ipynb) [Tutorial 1](notebooks/tutorials/tutorial_1_expressing_a_goal_as_an_algorithm.ipynb): Expressing a user experimental goal as an algorithm and executing the algorithm on the true underlying function. 

2) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/src47/multibax-sklearn/blob/main/notebooks/tutorials/tutorial_2_algorithm_execution_surrogate_model.ipynb) [Tutorial 2](notebooks/tutorials/tutorial_2_algorithm_execution_surrogate_model.ipynb): Executing an algorithm on posterior draws from a trained surrogate Gaussian process model. 

3) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/src47/multibax-sklearn/blob/main/notebooks/tutorials/tutorial_3_defining_metrics.ipynb) [Tutorial 3](notebooks/tutorials/tutorial_3_defining_metrics.ipynb): Defining metrics to quantify data acquisition quality: Number Obtained and Posterior Jaccard Index.

4) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/src47/multibax-sklearn/blob/main/notebooks/tutorials/tutorial_4_data_acquisition_using_BAX.ipynb) [Tutorial 4](notebooks/tutorials/tutorial_4_data_acquisition_using_BAX.ipynb): Using BAX strategies (InfoBAX, MeanBAX and SwitchBAX) to find a "wishlist" of regions in a magnetic alloys dataset. 

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
