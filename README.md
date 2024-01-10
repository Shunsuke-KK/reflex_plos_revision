# Overview

This page contains files related to the paper, "Identifying essential factors for energyefficient walking control across a wide range of velocities in reflex-based musculoskeletal systems" 
(https://doi.org/10.1371/journal.pcbi.1011771).

*  **assets**: musculoskeletal models is contained
*  **cmaes**: cmaes (optimizing algorithm) codes. I used (https://pypi.org/project/cmaes/), and some functions were added for this study.
*  **custom_env**: This folder contain gym-like environment for musculoskeletal model. Reset function, step function, fall down detect function, sensory function are included in the environment.
*  **func_data**: Polynimial functions of control parameters are contained.
*  **graph**: The codes and graphs in the paper are here.
*  **pic**: snapshots for different target velocities are contained.
*  **reflex_opt**: This contains main codes for optimizing parameters and running the result.

# Optimize control parameters
