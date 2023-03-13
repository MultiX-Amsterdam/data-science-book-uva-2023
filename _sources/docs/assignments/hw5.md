# Assignment 5: PyTorch Basics

(Last updated: Mar 13, 2023)

Your task in this assignment is to implement a deep neural network for the Smell Pittsburgh dataset in predicting the presence of bad odors using air quality data.

Create an empty notebook on your local machine or Google Colab to start the assignment.
To prepare and have the dataset ready, check the [Structured Data Processing Tutorial](../modules/structured-data-module/tutorial-structured-data).
Make sure that you use the previous 3 hours of data to predict the presence of smell in the future 8 hours.
This means you need to set `b_hr_sensor=2` and `f_hr_smell=7` in the `compute_feature_label` function (as shown in the tutorial notebook above).

Use the knowledge that you learned from self-studying the [PyTorch Basics](../lectures/lec9) to build the PyTorch pipeline.
You can use any type of network to complete this assignment.
However, we suggest starting by implementing a simple logistic regression neural network using one hidden layer with 64 hidden units.
This means using the sigmoid activation function with the cross-entropy loss.
We encourage you to gradually add complexity to the neural network once you finish the suggested model above.

Do not worry too much about the model performance, as it is hard to reach a good performance in this task.
But we encourage you to tune the model to have as high performance as possible.
When in doubt about the basic concepts of deep learning, check the [deep learning overview lecture](../lectures/lec5).
You can work in groups and discuss the assignment with others.

## Additional Recourses

Check the [PyTorch website](https://pytorch.org/tutorials/) for more resources during implementation.
