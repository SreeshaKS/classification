# <div align="center"> CS B551 - Assignment 4: Machine Learning
####  <div align="center"> CSCI B551 - Elements of Artificial Intelligence

<br>

###### Name: Sreesha Srinivasan Kuruvadi
###### Email: *sskuruva@iu.edu*
<br>

***
### [Part 1: K-Nearest Neighbors Classification](https://github.iu.edu/cs-b551-fa2021/sskuruva-a4/tree/master/k_nearest_neighbors.py)
***

#### Problem Statement:
Your goal in this part is to implement a k-nearest neighbors classifier from scratch. Your GitHub repository contains the skeleton code for two files that will be used to implement the algorithm: utils.py and k_nearest_neighbors.py.

KNearestNeighbors object, the following parameters must be specified (or their default values will be used):

- n_neighbors: the number of neighbors a sample is compared with when predicting target class values
(analogous to the value k in k-nearest neighbors).

- weights: represents the weight function used when predicting target class values (can be either ‘uniform’ or ‘distance’). 
Setting the parameter to ‘distance’ assigns weights proportional to the inverse of the distance from the test sample to each neighbor.

- metric: represents which distance metric is used to calculate distances between samples. There are two options: ‘l1’ or ‘l2’, which refer to the Manhattan distance and Euclidean distance respectively.
#### Command:
<code>  python3 multilayer_perceptron.py knn </code>


### [Part 2: Multilayer Perceptron Classification](https://github.iu.edu/cs-b551-fa2021/sskuruva-a4/tree/master/multilayer_perceptron.py)

***

#### Problem Statement:
Your goal in this part is to implement a feedforward fully-connected multilayer perceptron classifier with one hidden layer (as shown in the description above) from scratch. As before, your GitHub repository contains the skeleton code for two files that will be used to implement the algorithm: utils.py and multilayer_perceptron.py.

The multilayer_perceptron.py file defines the MultilayerPerceptron class that we will use to implement the algorithm from scratch. Just like the previous part, the __init__ function has already been properly implemented for you. The attributes for the class itself are described in detail in the skeleton code. When creating the MultilayerPerceptron object, the following parameters must be specified (or their default values will be used):
• n_hidden: the number of neurons in the one hidden layer of the neural network.
• hidden_activation: represents the activation function of the hidden layer (can be either ‘identity’,
‘sigmoid’, ‘tanh’, or ‘relu’).
• n_iterations: represents the number of gradient descent iterations performed by the fit(X, y)
method.
• learning_rate: represents the learning rate used when updating neural network weights during gra- dient descent.
#### Command:
<code>  python3 multilayer_perceptron.py mlp </code>