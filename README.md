# Targeted materials discovery using Bayesian Algorithm Execution (sklearn-bax)

This repo applies the Bayesian Algorithm Execution framework to the task of targeted and precise materials discovery. Users state their experimental goal as a simple algorithm which is automatically converted into a goal-align, intelligent data acquisition strategy. This framework applies to multiproperty measurements and goes beyond the capabilities of traditional multiobjective bayesian optimization algorithms. 


<img width="1103" alt="Screen Shot 2023-11-17 at 8 11 07 PM" src="https://github.com/src47/sklearn-bax/assets/39596225/beeb53e1-bbe6-47c8-89a4-fefb510143a9">

## Installation

1) Make a new local folder and clone the repository

```
git clone https://github.com/src47/sklearn-bax.git
```

2) Install requirements

```
pip install -r requirements.txt
```

## Directories

**src** 

Contains the source code for the project:
- src/acquisition: implementation of the three acquisition functions (InfoBAX, MeanBAX and SwitchBAX)
- algorithms.py, helper_subspace_functions.py: implementation of various experimental goals as algorithms 
- metrics.py: implement of the n_obtained and jaccard_posterior_index metrics 
- utils.py: misc
- plotting.py: visualization code 

**datasets** 

For ternary phase diagram data (cite), please use the following instructions:
- Download the data file "FeCoNi_benchmark_dataset_220501a.mat" from https://github.com/usnistgov/remi/tree/nist-pages/data/Combinatorial%20Libraries/Fe-Co-Ni and place it in the datasets directory. 

**notebooks** 

We highly recommend reviewing the following tutorial notebooks before using the method. 

1) tutorial_1_expressing_a_goal_as_an_algorithm.ipynb: This notebook explains how to pose an experimental goal as an algorithm and showcases running an algorithm on the ground-truth function as well as running an algorithm on posterior draws from a trained surrogate Gaussian Process model. 

2) tutorial_2_defining_metrics.ipynb: This notebook explains the two metrics (n_obtained and posterior_jaccard_index) used to assess the quality of data acquisition. 

3) tutorial_3_data_acquisition_using_BAX.ipynb: This notebook demonstrates using InfoBAX, MeanBAX and SwitchBAX to find a wishlist of regions in a ternary phase diagram. 

**Please direct any questions or comments to chitturi@stanford.edu, akashr@stanford.edu or wdn@stanford.edu. 
