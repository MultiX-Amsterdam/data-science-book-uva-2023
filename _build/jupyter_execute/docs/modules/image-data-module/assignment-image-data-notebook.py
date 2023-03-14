#!/usr/bin/env python
# coding: utf-8

# # Garbage Classification - A task from the Gemeente
# 
# (Last updated: Mar 14, 2023)[^credit]
# 
# [^credit]: Credit: this teaching material is created by Bryan Fleming under the supervision of [Yen-Chia Hsu](https://github.com/yenchiah).
# 
# Here is an online version of [this notebook in Google Colab](https://colab.research.google.com/drive/10kYsTaWb7DA-qvroa73PaLdbSpAfeckD?usp=sharing). This online version is just for browsing. To work on this notebook, you need to copy a new one to your own Google Colab.

# In[ ]:


# Importing Libraries

import math
import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
from torchvision.utils import make_grid
from IPython.display import Image
import base64
import matplotlib.pyplot as plt
from IPython.display import HTML


# **First, you need to enable GPUs for the notebook in Google Colab:**
# 
# - Navigate to Edit→Notebook Settings
# - Select GPU from the Hardware Accelerator drop-down

# In[ ]:


if torch.cuda.is_available():
    device = torch.device("cuda")          # use CUDA device
else:
    device = torch.device("cpu")           # use CPU device
device


# Make sure that the ouput from the above cell is `device(type='cuda')`. If you see `device(type='cpu')`, it means that you did not enable GPU usage on Google Colab. Go to [this page](https://web.eecs.umich.edu/~justincj/teaching/eecs442/WI2021/colab.html) and check the "Use GPU as an accelerator" part for details. Please be patient for this tutorial and the assignment, as training neural networks for Computer Vision tasks typically takes a lot of time.

# ## Scenario

# As members of the data science course, you have been tasked with a challenging and meaningful project: you will help give feedback on machine learning algorithms, designed for classifying types of garbage for the Gemeente Amsterdam. This project offers a unique opportunity to apply your data science skills to a real-world problem, making a positive impact on your local community.
# 
# Waste management is a critical issue that affects us all. As our population grows (and we know how much the population of Amsterdam is growing) and consumption increases, the amount of waste we generate also increases. One way to manage this waste is to classify it according to its type, making it easier to handle and recycle. However, manually classifying waste is time-consuming and prone to error, which is where machine learning can help.
# 
# Machine learning algorithms can be trained to recognize patterns in data and make accurate predictions based on those patterns. In the case of garbage classification, machine learning can help to identify the type of garbage based on its physical properties, such as size, shape, and color. By automating this process, we can improve the efficiency and accuracy of waste management and recycling.
# 
# As you work on this project, you will have the opportunity to apply your knowledge of data science, statistics, and machine learning to a real-world problem. You will learn how to clean, preprocess, and explore data, how to train machine learning algorithms, and how to evaluate and optimize the performance of your models.
# 
# This project will challenge you to think creatively, to collaborate with your peers, and to apply your skills to a problem that has real-world implications. We encourage you to take ownership of your work, to explore different methods and algorithms, and to document your process and results thoroughly.
# 
# In the end, your efforts will contribute to improving the waste management and recycling practices of the Gemeente Amsterdam, making a positive impact on the environment and the community. We wish you the best of luck in this exciting and meaningful project, and we look forward to seeing the results of your hard work.
# 
# The Gemeente needs critical analysis and feedback on their draft algorithms, so they can make an effective classifcation system in the future - to replace most of the current manpower and update the convoluted systems currently in place.

# Assignment 3
# - The loss when using the SGD optimizer drops slowly, and eventurally it may still be able to reach a good performance. But when we change the optimizer to Adam, we start to get a boost in model performance and a faster decrease in the loss.
# 
# Assignment 4
# - When we use a very small learning rate, the loss almost does not change, and the performance of the model changes only a little bit (but is still changing). When we use a very large learning rate, we see that the loss and the model performance oscillate (i.e., alternating between some low and high values).
# 
# Assignment 5
# - When we initialize all the weights to zero, we see that the loss and model performance almost never change. It looks like the model just stopped working. However, when we change the weight initialization to the Kaiming method, we see that the model can now be trained significantly faster than all the other settings when using the SGD optimizer with the same learning rate.
# 
# Assignment 6
# - We recommend using ResNet18 and a single linear layer at the end for classification. The recommended optimizer is Adam, the suggested learning rate is 5e-5, and the suggested number of epoch is 15. We also suggest using the Kaiming weight initialization method. These hyperparameters were found to produce the highest accuracy on the validation set while also avoiding overfitting. It is recommended that the Gemeente use this network with these hyperparameters as a starting point for further development of the AI garbage classifier. Some inspiration and functions accredited to [this Kaggle page](https://www.kaggle.com/code/aadhavvignesh/pytorch-garbage-classification-95-accuracy/notebook?scriptVersionId=38278889).
# 
# Assignment 7.1
# - The last layer was changed so it 'collapses' from the previous layer of the network's last fully connected layer to the number of classes in our model, so we can actually make predictions.
# 
# Assignment 7.2
# - Tis issue is the network failing to break symmetry, also known as the "symmetry problem," is a well-known problem in deep learning. When we have multiple neurons in a layer with the same weights and biases, they will produce the same output, and the gradients will be the same for all the neurons. This makes it impossible for the network to learn different features from the data, as all the neurons in a layer will contribute equally to the output.
# - By randomly initializing the weights, we break the symmetry, and each neuron will learn different features from the data. This is because the random initialization ensures that each neuron in a layer has a different starting point and is optimized differently, resulting in a diverse set of learned features. This allows the network to learn complex representations of the data, leading to better performance.
# 
# Assignment 7.3
# - Zero initialization: initializes all weights to zero. This can cause problems with symmetry breaking and cause all neurons to update identically during training, leading to poor performance.
# - Gaussian distributed initialization: initializes weights with random values drawn from a normal distribution. This method can help to break symmetry and enable better training, but the variance of the distribution needs to be carefully chosen to avoid exploding/vanishing gradients.
# - Kaiming initialization: similar to Xavier initialization, but designed specifically for ReLU activation functions that can cause problems with dying ReLUs if not initialized properly. This method scales the weights based only on the number of input neurons, rather than both input and output neurons as in Xavier initialization.
# - [EXTRA] Xavier initialization: scales the weights based on the number of input and output neurons, helping to ensure that the variance of the activations remains roughly constant across layers. This can lead to faster convergence and better performance, particularly for tanh activation functions.
# - More information about weight initialization can be found in [this notebook](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial4/Optimization_and_Initialization.html).
# 
# Assignment 7.4
# - The phenomenon where adding more layers to a neural network leads to worse performance is known as the "overfitting problem". This occurs when a model is too complex and fits the training data too closely, leading to poor generalization to new data. To fix this issue, we can use regularization methods such as L1, L2, and dropout. L1 regularization involves adding a penalty to the loss function that encourages weights to be sparse (so close to 0, they essentially act as a 0 weight would), while L2 regularization adds a penalty that encourages small weights. Dropout randomly "drops out" neurons during training, which helps prevent overfitting. By using these regularization techniques, we can help ensure that the model generalizes well to new data, while still keeping the architecture mostly the same.
# 
# Assignment 7.5
# - When using a high learning rate, the model's optimization algorithm may overshoot the optimal weights and biases during training, leading to instability and divergence. This can result in the loss function not converging or even increasing. To address this issue, we can use techniques such as learning rate scheduling, which gradually reduces the learning rate over time, or we can use adaptive optimization algorithms such as Adam, which dynamically adjust the learning rate during training based on the gradients. Additionally, regularization techniques such as dropout and weight decay can help to prevent overfitting and improve generalization.
# - Conversely, when using a very small learning rate, the optimizer will only take very small steps in updating the model parameters, which means that it will take a very long time to train our model.
# 
# Assignment 7.6
# - ReLU activation function - ReLU has a derivative of 1 for all positive inputs, which helps prevent the gradient from becoming too small.
# - Initialization techniques - Properly initializing the weights can help prevent the gradient from becoming too small or too large.
# - Dropout - Dropout randomly "drops out" neurons during training, which helps prevent overfitting and can also prevent the gradient from becoming too small.
# - L1 and L2 regularization - L1 regularization adds a penalty term to the loss function that encourages sparse weight matrices, while L2 regularization adds a penalty term that encourages small weights. This helps prevent the gradient from becoming too large.
# - These methods work by either ensuring that the gradient doesn't become too small or too large, or by preventing overfitting, which can exacerbate the vanishing gradient problem.
# - [EXTRA] Batch Normalization - Normalizing the input to each layer of the network can help prevent the gradient from becoming too small or too large.
# - [EXTRA] Gradient Clipping - Limiting the maximum or minimum value of the gradients can prevent the gradients from becoming too large or too small.
# - [EXTRA] Residual connections - Adding residual connections to the network can help prevent the gradient from becoming too small as the signal can bypass the problematic layers. This solution corresponds to the ResNet that we used in this tutorial.
# - More information about Vanishing Gradient can be found from [this notebook](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial3/Activation_Functions.html).

# ## Load Datasets

# **These aren't mock-up datasets, they're real world data!**

# To work with the [garbage classification dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification), we need to download the data. We have prepared a [GitHub repository](https://github.com/MultiX-Amsterdam/image-data-module/tree/main/garbage-classification-dataset) with the dataset for you.
# 
# The following code checks if you are using Google Colab or on a local machine with GPU. If you clone the repository to your local machine, there is no need to download the dataset. However, if you are on Google Colab, the code cell below will clone the dataset to the Google Colab's file disk.

# In[ ]:


# This is the relative path for data if you clone the repository to a local machine
data_dir  = 'garbage-classification-dataset/garbage-classification-images'
try:
  classes = os.listdir(data_dir)
  print(classes)
except:
  # This means that we are not on a local machine, so we need to clone the dataset
  get_ipython().system('git clone https://github.com/MultiX-Amsterdam/image-data-module')
  # Below is the path of the data on the Google Colab disk
  data_dir  = '/content/image-data-module/garbage-classification-dataset/garbage-classification-images'
  classes = os.listdir(data_dir)
  print(classes)


# ### Transform Data

# To input the data into a machine learning algorithm, we need to transform the dataset into a normalized tensor. We can do this using PyTorch's built-in transforms module.

# In[ ]:


from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor()])


# In this example, we're resizing the images to 256x256 pixels and cropping them to 224x224 pixels from the center. We then convert the images to tensors using transforms.ToTensor().
# 

# In[ ]:


dataset = ImageFolder(data_dir, transform = transformations)


# We then apply these transforms to the dataset using the datasets.ImageFolder class, which automatically applies the transforms to the images in the dataset:

# ### Visualize Data

# As we work on building our machine learning model for image classification, it's important to understand the data that we're working with. One way to gain a better understanding of the data is by visualizing it. By looking at the images in the dataset, we can get a sense of what features and patterns are present in the images, which can inform our choice of model architecture and training strategy.
# 
# To visualize the data, we can use a Python library like matplotlib to display sample images from the dataset along with their corresponding labels. This can help us get a sense of what types of images are present in the dataset, and how they are distributed across different classes.
# 
# To make this process easier, we can define a function that takes in a dataset object and displays a grid of images along with their corresponding labels. This function can be called at any point during our analysis to visualize the data and gain a better understanding of the images we're working with.
# 
# As we continue to work on building our machine learning model, we'll be using this function to visualize the data and its classes, and gain insights that will help us build a better model.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

def show_sample(img, label):
    print("Label:", dataset.classes[label], "(Class No: "+ str(label) + ")")
    plt.imshow(img.permute(1, 2, 0))


# In[ ]:


img, label = dataset[2]
show_sample(img, label)


# ### Split Data
# 
# In order to train our machine learning model, we need to split our data into training, validation, and test sets. We need to set a seed for the random number generator. This ensures that our data is split in a consistent way each time we run our code, which is important for reproducibility. In the code snippet below, we set the random seed to 23 and the manual seed to ensure that the split is consistent.

# In[ ]:


#Setting a seed for the random split of data:

random_seed = 23
torch.manual_seed(random_seed)


# Once the seed is set, we can split our dataset into training, validation, and test sets. In the code snippet below, we use PyTorch's random_split function to split our dataset into three sets: 1593 images for training, 176 images for validation, and 758 images for testing.

# In[ ]:


# Splitting the data:

train_ds, val_ds, test_ds = random_split(dataset, [1593, 176, 758])
len(train_ds), len(val_ds), len(test_ds)


# ### Build a Dataloader
# 
# Finally, we can load our dataset into PyTorch data loaders, which will allow us to efficiently feed the data into our machine learning model. In the code snippet below, we define data loaders for our training and validation sets, with a batch size of 32 for the training data and a batch size of 64 for the validation data. We also set the shuffle, num_workers, and pin_memory parameters to optimize the data loading process.
# 
# Keep in mind that the batch size is a hyperparameter that you can tune. However, setting the batch size to a large number may not be a good idea if your GPU and computer only have a small computer memory. Using a larger batch size consumes more computer and GPU memory.

# In[ ]:


# Loading the data

batch_size = 32

train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers = 4, pin_memory = True)


# ### Check the Dataloader

# As we mentioned earlier, visualizing our data is an important step in the machine learning process. By looking at sample images from our dataset, we can gain insights into the structure of the data and identify any issues with preprocessing or data loading.
# 
# To make the visualization process easier, we can define a function called show_batch that takes in a PyTorch data loader and displays a grid of images from the batch along with their corresponding labels. This can help us identify patterns and features in the images, and ensure that our data is being loaded and preprocessed correctly.
# 
# In the code snippet below, we define the show_batch function. This function takes in a data loader object and displays a grid of images from the batch along with their corresponding labels. We use the make_grid function to create a grid of images from the batch, with 16 images per row. The permute function is used to change the order of the dimensions of the tensor to match the expected input format for imshow.
# 
# 

# In[ ]:


# As before, we have a simple tool for visualization

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow = 16).permute(1, 2, 0))
        break


# In[ ]:


show_batch(train_dl)


# Now that we've had a chance to visualize our data and ensure that our data loading and preprocessing is working correctly, it's time to move on to the main part of our machine learning implementation: network creation.
# 
# Our goal is to build a neural network that can accurately classify images into different categories. To do this, we'll need to create a network architecture that is appropriate for the task at hand. This can involve a variety of decisions, such as the number of layers in the network, the size of the filters used in convolutional layers, and the activation functions used throughout the network.
# 
# Building a neural network can be a complex process, but fortunately, we have powerful tools and libraries available to simplify the task. We'll be using PyTorch, a popular deep learning framework, to build our network.

# ## Build the Pipeline

# In order to train and evaluate our neural network, we'll need to be able to load our pre-processed data into our model and assess its performance on our validation and test sets. To do this, we have two critical functions at our disposal: data loading and model evaluation.
# 
# Data loading involves reading our pre-processed data from disk and converting it into a format that can be input into our neural network. This can involve operations such as loading images, transforming them into tensors, and creating batches for efficient processing. By properly loading our data, we can ensure that our network is able to process the data effectively and accurately.
# 
# Model evaluation, on the other hand, involves assessing the performance of our neural network on our validation and test sets. We'll be using a variety of metrics, such as accuracy and F1 score, to evaluate the performance of our model. This is critical for determining how well our network is able to classify images and whether it is ready for deployment.

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ImageClassificationBase(nn.Module):
    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = self.accuracy(out, labels)      # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        
    return history


# After completing the initial data wrangling and evaluation, we're now ready to move onto the exciting part of our machine learning implementation: tweaking and evaluating our neural networks.
# 
# We have several different neural networks that we'll be working with, all based on the ResNet50 architecture, but with key differences. Each network has been created and initialized with specific goals and objectives in mind. While some of the methods used may be exaggerated for the purposes of demonstrating key concepts, it's important to keep in mind that each of these methods plays a critical role in the overall performance of our network.
# 
# During our evaluation process, we'll be focusing primarily on accuracy as our key metric for assessing the performance of our neural networks. By analyzing the accuracy of our networks, we can determine how well they are able to classify images and identify areas where they may be struggling. While there are other metrics, such as F1 score and confusion matrices, that can also provide valuable insights into the performance of our networks, we'll be primarily focusing on accuracy to streamline our evaluation process.
# 
# Throughout this process, you'll be asked a series of questions about our neural networks and their performance. By fully engaging with this process, you'll gain a deeper understanding of the key concepts and techniques involved in building and evaluating machine learning models. So let's dive in and start building our networks!
# 
# As you work through our implementation, you may (hopefully) notice that some of the features of our neural networks are exaggerated or skewed for specific goals. For example, we may use extreme weight initialization techniques to emphasize the importance of proper weight initialization, or to allow the network to display certain traits. While these features may not always be representative of real-world scenarios, they can be helpful in highlighting key concepts and techniques that are critical for success in machine learning. By understanding the reasoning behind these exaggerated features, you'll be better equipped to apply these techniques in real-world situations and achieve better performance from your neural networks.
# 
# Below you'll see the Network Architecture for ResNet50 - It's interesting to see the complex detail behind recognizing features, and determining their importance for a classification problem, we will be using tweaked versions of this CNN for our CNN's. The ResNet architecture is explained in the following paper:
# - [He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).](https://arxiv.org/pdf/1512.03385.pdf)
# 
# Notice that in this tutorial, to save computational time, we are going to use the ResNet18 structure, which is a smaller version of the ResNet.

# ![](https://github.com/MultiX-Amsterdam/image-data-module/blob/main/images/resnet50.png?raw=true)
# 
# Image source -- https://doi.org/10.3390/s20020447

# ### Task 3: Optimizer
# 
# Now let us create a basic ResNet for this task.
# 
# In the code below, we first use `get_default_device()` to get the default device (CPU or GPU) for training. We then initialize our `ResNetGaussianWeight` model and move it to the device using `to_device()`. Next, we move our training and validation data loaders to the device using `DeviceDataLoader()`. We set the number of epochs using `num_epochs`, and select the optimizer and learning rate using `opt_func` and `lr`, respectively. Finally, we train the model using the `fit()` function and store the training history in history.
# 
# It is worth noting that this is the only time we will be breaking down this specific block of code in such detail. From here on out, we will assume a basic understanding of the concepts and functions used in this code and focus on the unique features and characteristics of each neural network we explore.

# In[ ]:


class ResNetGaussianWeight(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet18(pretrained=False)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))
        
        # Gaussian random weight initialization for all the weights
        for module in self.network.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                init.normal_(module.weight, mean=0, std=0.0001)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, xb):
        return self.network(xb)


device = get_default_device() # get the default device for training
model = ResNetGaussianWeight() # initialize the ResNet model
model = to_device(model, device) # move the model to the device (CPU or GPU)
train_dl = DeviceDataLoader(train_dl, device) # move the training data loader to the device
val_dl = DeviceDataLoader(val_dl, device) # move the validation data loader to the device

num_epochs = 10 # set the number of epochs
opt_func = torch.optim.SGD # set the optimizer
lr = 1e-4 # set the learning rate

history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func) # train the model and store the training history


# The code above uses the [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) (Stochastic Gradient Descent) optimizer. Now let us change the optimizer to [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html), which is explained in the following paper:
# - [Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.](https://arxiv.org/pdf/1412.6980.pdf)

# In[ ]:


device = get_default_device() # get the default device for training
model = ResNetGaussianWeight() # initialize the ResNet model
model = to_device(model, device) # move the model to the device (CPU or GPU)
train_dl = DeviceDataLoader(train_dl, device) # move the training data loader to the device
val_dl = DeviceDataLoader(val_dl, device) # move the validation data loader to the device

num_epochs = 10 # set the number of epochs
opt_func = torch.optim.Adam # set the optimizer
lr = 1e-4 # set the learning rate

history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func) # train the model and store the training history


# #### Assignment 3
# 
# In this task, we used two different optimizers while keeping all other settings and hyperparameters the same. What do you observe for the evaluation results? What are the differences?
# 
# YOUR ANSWER HERE

# ### Task 4: Learning Rate
# 
# Continue Task 3, now let us keep using the SGD optimizer while lowering the learning rate.

# In[ ]:


device = get_default_device() # get the default device for training
model = ResNetGaussianWeight() # initialize the ResNet model
model = to_device(model, device) # move the model to the device (CPU or GPU)
train_dl = DeviceDataLoader(train_dl, device) # move the training data loader to the device
val_dl = DeviceDataLoader(val_dl, device) # move the validation data loader to the device

num_epochs = 10 # set the number of epochs
opt_func = torch.optim.SGD # set the optimizer
lr = 1e-8 # set the learning rate

history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func) # train the model and store the training history


# Now let us increase the learning rate to a large number and still keep using the SGD optimizer.

# In[ ]:


device = get_default_device() # get the default device for training
model = ResNetGaussianWeight() # initialize the ResNet model
model = to_device(model, device) # move the model to the device (CPU or GPU)
train_dl = DeviceDataLoader(train_dl, device) # move the training data loader to the device
val_dl = DeviceDataLoader(val_dl, device) # move the validation data loader to the device

num_epochs = 10 # set the number of epochs
opt_func = torch.optim.SGD # set the optimizer
lr = 12 # set the learning rate

history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func) # train the model and store the training history


# #### Assignment 4
# 
# In this task, we used two different learning rates while keeping all other settings and hyperparameters the same. What do you observe for the evaluation results? What are the differences of these results when compared to the SGD optimizer version in Task 3?
# 
# YOUR ANSWER HERE

# ### Task 5: Weight Initiation
# 
# Next, let us initialize all the model weights to zero. We still keep using the SGD optimizer and keep the learning rate the same as the one used in Task 3.

# In[ ]:


class ResNetZeroWeight(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet18(pretrained=False)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))
        
        # Zero weight initialization for all the weights
        for module in self.network.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                init.zeros_(module.weight)
                if module.bias is not None:
                      init.zeros_(module.bias)

    def forward(self, xb):
        return self.network(xb)


device = get_default_device() # get the default device for training
model = ResNetZeroWeight() # initialize the ResNet model
model = to_device(model, device) # move the model to the device (CPU or GPU)
train_dl = DeviceDataLoader(train_dl, device) # move the training data loader to the device
val_dl = DeviceDataLoader(val_dl, device) # move the validation data loader to the device

num_epochs = 10 # set the number of epochs
opt_func = torch.optim.SGD # set the optimizer
lr = 1e-4 # set the learning rate

history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func) # train the model and store the training history


# Besides the Gaussian weight initialization that we used in the previous task, we can also use uniform weight initialization. The method is documented in the following paper and has been used widely.
# - [He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision (pp. 1026-1034).](https://arxiv.org/pdf/1502.01852.pdf)
# 
# In the following code, we change only the weight initialization. We still keep using the SGD optimizer and keep the learning rate the same as the one used in Task 3.
# 

# In[ ]:


class ResNetUniformWeight(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet18(pretrained=False)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))
        
        # Kaiming initialization for all the weights
        for module in self.network.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in)
                    init.uniform_(module.bias, -bound, bound)

    def forward(self, xb):
        return self.network(xb)


device = get_default_device() # get the default device for training
model = ResNetUniformWeight() # initialize the ResNet model
model = to_device(model, device) # move the model to the device (CPU or GPU)
train_dl = DeviceDataLoader(train_dl, device) # move the training data loader to the device
val_dl = DeviceDataLoader(val_dl, device) # move the validation data loader to the device

num_epochs = 10 # set the number of epochs
opt_func = torch.optim.SGD # set the optimizer
lr = 1e-4 # set the learning rate

history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func) # train the model and store the training history


# #### Assignment 5
# 
# This task shows you two other different ways of initializing the model weights (i.e., the model parameters). What do you observe for the evaluation results? What are the differences of these results when compared to the SGD optimizer version in Task 3?
# 
# YOUR ANSWER HERE

# ### Task 6: Experiments
# 
# Now, given all of this, it is up to you to analyse these algorithms, the outcomes, and determine the variables which contribute more; the Gemeente is relying on you for the future, so do your best to understand and advise based on this.

# #### Assignment 6
# 
# Conduct experiments and write a short paragraph to recommend the best CNN, optimizer, and learning rate (of the ones used) for the Gemeente to carry forward in their development of a full pipeline AI garbage classifier. You can try [different network architecture](https://pytorch.org/vision/main/models.html#classification), number of epochs, learning rate, and weight initializations.

# YOUR ANSWER HERE

# ### Task 7: Theory Questions

# #### Assignment 7.1: Role of the Last Layer
# 
# We modified the ResNet architecture to replace only the last layer for our task. The rest of the neural net remains the same as we imported the network. What is the purpose of changing the last layer of the network?

# YOUR ANSWER HERE

# #### Assignment 7.2: Constant Weight Initialization
# 
# As you saw in Task 5, if we were not to randomize the model weights and instead initialize the weights to a constant value (or simply zero), we would encounter a famous problem of Deep Learning. Research online to discover what we call this issue, why it happens, and why randomly initializing the weights fixes this issue.

# YOUR ANSWER HERE

# #### Assignment 7.3: Weight Initialization Comparison
# 
# There are different types of weight initializations. You saw Zero, Gaussian Distributed, and Kaiming methods here. Research these methods, and try to explain briefly in your own words the strengths and weaknesses of each.

# YOUR ANSWER HERE

# #### Assignment 7.4: Network Complexity
# 
# You might think that a Deep Neural Network is always better. But as you saw in the tutorial, sometimes adding more layers to increase the complexity of neural networks can completely miss the point and result in worse performance. There is a famous name for this phenomenon. Find out what it is called and briefly describe why it happens and how to fix it if the architecture is to remain mostly the same.

# YOUR ANSWER HERE

# #### Assignment 7.5: Learning Rate
# 
# In Task 4, we used very low and high learning rates. This is a classic problem in Deep Learning. Identify what arises when we do this, and give your best resolutions to these issues (some of which we used).

# YOUR ANSWER HERE

# #### Assignment 7.6: Vanishing Gradient
# 
# Although we did not show you here, neural networks can suffer from Vanishing Gradient. Mathematically it is because their derivative is non-linear, so in the back-propagation stage, the gradient of the loss function compared to the weights is too small. They cannot update the weights enough to make meaningful changes. There are many ways to fix these issues. Try to list as many of them as possible, and using the above information and your own further research, try to convey why these fix the issue.

# YOUR ANSWER HERE
