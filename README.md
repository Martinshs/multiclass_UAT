# Neural Network Classification Without Training

In this repository, we present code that can solve any binary classification problem using a deep neural network of the multilayer perceptron (MLP) type, without the need to solve any optimization problem.

# Description

In [1](https://arxiv.org/abs/2407.07556), it is proven constructively that an MLP with width 2 (and a number of layers depending on the number of data points) can solve any classification problem. In other words, the MLP is simultaneously controllable or, alternatively, satisfies finite sample memorization. In [1](https://arxiv.org/abs/2407.07556), to construct the parameters, it is not necessary to define any optimization problem. Moreover, these parameters depend solely on the dataset (data and labels).

Motivated by this, in this repository, we introduce an algorithm that defines the parameters introduced in [1](https://arxiv.org/abs/2407.07556), to solve any classification problem.

In particular, the first observation in [1](https://arxiv.org/abs/2407.07556) is that the MLP parameters at each layer can be interpreted as hyperplanes. Furthermore, each time a hyperplane is defined, this introduces parameters for the neural network. Based on this idea, in [1](https://arxiv.org/abs/2407.07556), a neural network is inductively constructed from hyperplanes.

The code in this repository show how the hyperplanes (MLP parameters) interact simultaneously with the input data to map them to their respective labels (targets).

<p align="center">
    <img src="https://github.com/Martinshs/multiclass_UAT/edit/main/gifs/example_1.gif" alt="" /></a>
</p>


## Code Structure

- `utils.py`: Contains several functions. Among them are:
  * relu : ReLU function 
  * abline_2 : To plot the hyperplanes
  * obs_1d : 1d function considered as the obstacle for the 1d case.
  * gen_data :  Generate uniformly distributed data
  * check_folder : Creates the folder used to save the images. 
  * figure_names : Save the figures names (used to create gifs)
  * make_gif : Function that creates a gif.
              
  
- `auto_training.py`: This file contain the function `auto_train` which is used to create the paramters of the NN. This function need the data and labels of the problem.
- `feed_forward.py`:  This file contain the function `feedfoward_NN`. Given the input data and the parameter `feedfoward_NN` give the output of the MLP.
- `auto_binary_classification.ipynb`: Jupyter Notebooks with examples demonstrating the code.

## References
[1](https://arxiv.org/abs/2407.07556)  Deep Neural Networks: Multi-Classification and Universal Approximation, M. Hernandez, E. Zuazua.



