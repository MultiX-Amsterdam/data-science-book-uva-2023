# Preparation (Image Data Processing)

(Last updated: Mar 14, 2023)

This part will prepare you with the background knowledge that we will use for this module.
You will explore Google Teachable Machine and also prepare PyTorch basics.

## Task 1: Google Teachable Machine

In order to understand the inner workings of Computer Vision, it is important to first have a general overview of what it is.
We will use a much less robust dataset to show what image classification is.
First, download the [flowers recognition dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition).
You will need to make an account on Kaggle.

Next, go to the [Google Teachable Machine website](https://teachablemachine.withgoogle.com/).
Click get started to start creating an image project.
Upload each of the folders that classify the flower type as a class, and train your model.
If you do not know how Google Teachable Machine works, watch the videos in [this YouTube playlist](https://www.youtube.com/playlist?list=PLJfHZtseuscuTQfodmFnbZ3rBgCWsRT9t).

Train your model by picking the button.
Congratulations, you have trained your first Computer Vision Algorithm!
You can now export code based on a model you trained on data.
It will classify flowers as best as it can if they are from these species.
It will also try to classify flowers, not in these species, based on their appearance and shared features!

We will not use this model in the tutorial, as its output is a library (across many languages now) called TensorFlow, which is not what we will use in the assignment.
But this is a fun resource to play around with.
We encourage you to try and create other models.

## Task 2: PyTorch Basics

In the tutorial, we will use the [PyTorch deep learning framework](https://pytorch.org/).
You should already study the PyTorch basics from one of the lectures before.
If not, you should go through [this PyTorch tutorial notebook](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial2/Introduction_to_PyTorch.html) while at the same time watching the following two videos:
- [Introduction to PyTorch (Part 1)](https://www.youtube.com/watch?v=wnKZZgFQY-E)
- [Introduction to PyTorch (Part 2)](https://www.youtube.com/watch?v=schbjeU5X2g)

If you want to learn more about PyTorch, consider following another tutorial below:
- [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
