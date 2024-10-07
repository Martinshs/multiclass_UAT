# Neural Network Classification Without Training

In this repository, we present code that can solve any binary classification problem using a deep neural network of the multilayer perceptron (MLP) type, without the need to solve any optimization problem.

# Description

In [ARTICLE], it is proven constructively that an MLP with width 2 (and a number of layers depending on the number of data points) can solve any classification problem. In other words, the MLP is simultaneously controllable or, alternatively, satisfies finite sample memorization. In [ARTICLE], to construct the parameters, it is not necessary to define any optimization problem. Moreover, these parameters depend solely on the dataset (data and labels).

Motivated by this, in this repository, we introduce an algorithm that defines the parameters introduced in [ARTICLE], to solve any classification problem.

In particular, the first observation in [ARTICLE] is that the MLP parameters at each layer can be interpreted as hyperplanes. Furthermore, each time a hyperplane is defined, this introduces parameters for the neural network. Based on this idea, in [ARTICLE], a neural network is inductively constructed from hyperplanes.

The code above demonstrates how the hyperplanes (MLP parameters) interact simultaneously with the data to map them to their respective labels (targets).





## Repository Structure

- `utils.py`: Contains several functions. Among them are:
  * relu : ReLU function 
  * abline_2 : To plot the hyperplanes
  * obs_1d : 1d function considered as the obstacle for the 1d case.
  * gen_data :  Generate uniformly distributed data
  * check_folder : Creates the folder used to save the images. 
  * figure_names : Save the figures names (used to create gifs)
  * make_gif : Function that creates a gif.
              
  
- `solvers.py`: Contain the solvers. Namely,
  * solver_op :  Obstacle problem solver  ([inspired by DyCon-blog](https://deustotech.github.io/DyCon-Blog/tutorial/wp03/P0004))
  * sol_one_realization :  Solve one realization of the random minimizing movement
  * solver_avg_mb_op : Compute the average of a given number of realizations of the random minimizing movement
- `mini_batch_obstacle_problem.ipynb`: Jupyter Notebooks with examples demonstrating the code.

