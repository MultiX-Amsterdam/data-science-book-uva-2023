#!/usr/bin/env python
# coding: utf-8

# # Tutorial (Structured Data Processing)

# This tutorial will familiarize you with the data science pipeline of processing structured data, using a real-world example of building models to predict and explain the presence of bad smell events in Pittsburgh using air quality and weather data (as indicated in the following figure). The models are used to send push notifications about bad smell events to inform citizens, as well as to explain local pollution patterns to inform stakeholders.

# <img src="../../../assets/images/smellpgh-predict.png" style="max-width: 800px">

# The scenario is in the mext section of this tutorial, and more details is in the introduction section of the [Smell Pittsburgh paper](https://doi.org/10.1145/3369397). We will use the [same dataset as used in the Smell Pittsburgh paper](https://github.com/CMU-CREATE-Lab/smell-pittsburgh-prediction/tree/master/dataset/v1) as an example of structured data. During this tutorial, we will explain what the variables in the dataset mean and also guide you through model building.

# ## Scenario

# Local citizens in Pittsburgh are organizing communities to advocate for changes in air pollution regulations. Their goal is to investigate the air pollution patterns in the city to understand the potential sources related to the bad odor. The communities rely on the Smell Pittsburgh application (as indicated in the figure below) to collect smell reports from citizens that live in the Pittsburgh region. Also, there are air quality and weather monitoring stations in the Pittsburgh city region that provide sensor measurements, including common air pollutants and wind information.

# <img src="../../../assets/images/smellpgh-ui.png" style="max-width: 800px">

# You work in a data science team to develop models that can be suitable to map the sensor data to bad smell events. Your team has been working with the Pittsburgh local citizens closely for a long time, and therefore you know the meaning of each variable in the feature set that is used to train the machine learning model. The Pittsburgh community needs your help timely to analyze the data that can help them present evidence of air pollution to the municipality and explain the patterns to the general public.

# ## Tutorial Task 1

# We need to filter the unwanted geographical region. The geographical regions that we use is in Figure 6 in the Smell Pittsburgh paper.

# In[ ]:




