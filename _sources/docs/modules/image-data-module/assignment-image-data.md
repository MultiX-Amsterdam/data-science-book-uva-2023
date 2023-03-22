# Assignment (Image Data Processing)

(Last updated: Mar 14, 2023)

## Materials

- [Image Data Processing Assignment Online Notebook](assignment-image-data-notebook)

## Usage

Follow the steps below to set up the assignment:
- Use the Google Colab link at the begining of the notebook to set up your notebook on your Google Colab
- Work on the assignment on Google Colab

Follow the steps below to do the assignment:
- Complete the assignments that are indicated in the notebook. You can leave out the optional assignment if it is too difficult or takes too much time.
- For each task, try your best to implement the solution without checking the answer.
- If there are parts that you do not understand, feel free to ask questions during the work sessions or on Canvas.

## Answer

Below are the answers to the assignments. Do not check them before doing the assignments by yourself first.

:::{toggle}
Assignment 3
- The loss when using the SGD optimizer drops slowly, and eventurally it may still be able to reach a good performance. But when we change the optimizer to Adam, we start to get a boost in model performance and a faster decrease in the loss.

Assignment 4
- When we use a very small learning rate, the loss almost does not change, and the performance of the model changes only a little bit (but is still changing). When we use a very large learning rate, we see that the loss and the model performance oscillate (i.e., alternating between some low and high values).

Assignment 5
- When we initialize all the weights to zero, we see that the loss and model performance almost never change. It looks like the model just stopped working. However, when we change the weight initialization to the Kaiming method, we see that the model can now be trained significantly faster than all the other settings when using the SGD optimizer with the same learning rate.

Assignment 6
- We recommend using ResNet18 and a single linear layer at the end for classification. The recommended optimizer is Adam, the suggested learning rate is 5e-5, and the suggested number of epoch is 15. We also suggest using the Kaiming weight initialization method. These hyperparameters were found to produce the highest accuracy on the validation set while also avoiding overfitting. It is recommended that the Gemeente use this network with these hyperparameters as a starting point for further development of the AI garbage classifier. Some inspiration and functions accredited to [this Kaggle page](https://www.kaggle.com/code/aadhavvignesh/pytorch-garbage-classification-95-accuracy/notebook?scriptVersionId=38278889).

Assignment 7.1
- The last layer was changed so it 'collapses' from the previous layer of the network's last fully connected layer to the number of classes in our model, so we can actually make predictions.

Assignment 7.2
- Tis issue is the network failing to break symmetry, also known as the "symmetry problem," is a well-known problem in deep learning. When we have multiple neurons in a layer with the same weights and biases, they will produce the same output, and the gradients will be the same for all the neurons. This makes it impossible for the network to learn different features from the data, as all the neurons in a layer will contribute equally to the output.
- By randomly initializing the weights, we break the symmetry, and each neuron will learn different features from the data. This is because the random initialization ensures that each neuron in a layer has a different starting point and is optimized differently, resulting in a diverse set of learned features. This allows the network to learn complex representations of the data, leading to better performance.

Assignment 7.3
- Zero initialization: initializes all weights to zero. This can cause problems with symmetry breaking and cause all neurons to update identically during training, leading to poor performance.
- Gaussian distributed initialization: initializes weights with random values drawn from a normal distribution. This method can help to break symmetry and enable better training, but the variance of the distribution needs to be carefully chosen to avoid exploding/vanishing gradients.
- Kaiming initialization: similar to Xavier initialization, but designed specifically for ReLU activation functions that can cause problems with dying ReLUs if not initialized properly. This method scales the weights based only on the number of input neurons, rather than both input and output neurons as in Xavier initialization.
- [EXTRA] Xavier initialization: scales the weights based on the number of input and output neurons, helping to ensure that the variance of the activations remains roughly constant across layers. This can lead to faster convergence and better performance, particularly for tanh activation functions.
- More information about weight initialization can be found in [this notebook](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial4/Optimization_and_Initialization.html).

Assignment 7.4
- The phenomenon where adding more layers to a neural network leads to worse performance is known as the "overfitting problem". This occurs when a model is too complex and fits the training data too closely, leading to poor generalization to new data. To fix this issue, we can use regularization methods such as L1, L2, and dropout. L1 regularization involves adding a penalty to the loss function that encourages weights to be sparse (so close to 0, they essentially act as a 0 weight would), while L2 regularization adds a penalty that encourages small weights. Dropout randomly "drops out" neurons during training, which helps prevent overfitting. By using these regularization techniques, we can help ensure that the model generalizes well to new data, while still keeping the architecture mostly the same.

Assignment 7.5
- When using a high learning rate, the model's optimization algorithm may overshoot the optimal weights and biases during training, leading to instability and divergence. This can result in the loss function not converging or even increasing. To address this issue, we can use techniques such as learning rate scheduling, which gradually reduces the learning rate over time, or we can use adaptive optimization algorithms such as Adam, which dynamically adjust the learning rate during training based on the gradients. Additionally, regularization techniques such as dropout and weight decay can help to prevent overfitting and improve generalization.
- Conversely, when using a very small learning rate, the optimizer will only take very small steps in updating the model parameters, which means that it will take a very long time to train our model.

Assignment 7.6
- ReLU activation function - ReLU has a derivative of 1 for all positive inputs, which helps prevent the gradient from becoming too small.
- Initialization techniques - Properly initializing the weights can help prevent the gradient from becoming too small or too large.
- Dropout - Dropout randomly "drops out" neurons during training, which helps prevent overfitting and can also prevent the gradient from becoming too small.
- L1 and L2 regularization - L1 regularization adds a penalty term to the loss function that encourages sparse weight matrices, while L2 regularization adds a penalty term that encourages small weights. This helps prevent the gradient from becoming too large.
- These methods work by either ensuring that the gradient doesn't become too small or too large, or by preventing overfitting, which can exacerbate the vanishing gradient problem.
- [EXTRA] Batch Normalization - Normalizing the input to each layer of the network can help prevent the gradient from becoming too small or too large.
- [EXTRA] Gradient Clipping - Limiting the maximum or minimum value of the gradients can prevent the gradients from becoming too large or too small.
- [EXTRA] Residual connections - Adding residual connections to the network can help prevent the gradient from becoming too small as the signal can bypass the problematic layers. This solution corresponds to the ResNet that we used in this tutorial.
- More information about Vanishing Gradient can be found from [this notebook](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial3/Activation_Functions.html).
:::
