This is the self-supervised approach described in the thesis; in this stage, we apply multi-exponential models to generate $T_{2}$ decay signals. We generate the signals with either 2 or 3 parameters, where the last parameter corresponds to the cerebrospinal fluid. We then train the model with customized data and make inferences with the same dataset since this approach is considered self-supervised

This module consists of two main parts.
* The first corresponds to data generation with the two or three exponential models. Run ``` python data_generation.py ``` to generate a synthetic phantom. Feel free to change the path and the number of parameters of the compartments according to the needs. 
* The second part corresponds to training the model with the synthetically generated data. 
