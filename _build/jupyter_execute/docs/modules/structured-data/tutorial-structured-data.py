#!/usr/bin/env python
# coding: utf-8

# # Tutorial (Structured Data Processing)

# :::{warning}
# This page is under construction and not finished yet. Do not use it.
# :::

# (Last updated: Feb 11, 2023)

# This tutorial will familiarize you with the data science pipeline of processing structured data, using a real-world example of building models to predict and explain the presence of bad smell events in Pittsburgh using air quality and weather data (as indicated in the following figure). The models are used to send push notifications about bad smell events to inform citizens, as well as to explain local pollution patterns to inform stakeholders.

# \
# <img src="../../../assets/images/smellpgh-predict.png" style="max-width: 700px;">

# \
# The scenario is in the mext section of this tutorial, and more details is in the introduction section of the [Smell Pittsburgh paper](https://doi.org/10.1145/3369397). We will use the [same dataset as used in the Smell Pittsburgh paper](https://github.com/CMU-CREATE-Lab/smell-pittsburgh-prediction/tree/master/dataset/v1) as an example of structured data. During this tutorial, we will explain what the variables in the dataset mean and also guide you through model building.

# ## Scenario

# Local citizens in Pittsburgh are organizing communities to advocate for changes in air pollution regulations. Their goal is to investigate the air pollution patterns in the city to understand the potential sources related to the bad odor. The communities rely on the Smell Pittsburgh application (as indicated in the figure below) to collect smell reports from citizens that live in the Pittsburgh region. Also, there are air quality and weather monitoring stations in the Pittsburgh city region that provide sensor measurements, including common air pollutants and wind information.

# \
# <img src="../../../assets/images/smellpgh-ui.png" style="max-width: 700px;">

# \
# You work in a data science team to develop models to map the sensor data to bad smell events. Your team has been working with the Pittsburgh local citizens closely for a long time, and therefore you know the meaning of each variable in the feature set that is used to train the machine learning model. The Pittsburgh community needs your help timely to analyze the data that can help them present evidence of air pollution to the municipality and explain the patterns to the general public.

# ## Packages and Answers

# We put all the packages that are needed for this tutorial below:

# In[1]:


import pandas as pd
from os.path import isfile, join
from os import listdir


# The code block below contains answers for the assignments in this tutorial. **Do not check the answers in the next cell before practicing the tasks.**

# In[2]:


def check_answer_df(df_result, df_answer, n=1):
    """
    This function checks if two output dataframes are the same.
    """
    try:
        assert df_answer.equals(df_result)
        print("Test case %d passed." % n)
    except:
        print("Test case %d failed." % n)
        print("")
        print("Your output is:")
        print(df_result)
        print("")
        print("Expected output is:")
        print(df_answer)
        

def answer_preprocess_smell(df):
    """
    This function is the answer of task 4.
    """
    # Copy the dataframe to avoid editing the original one.
    df = df.copy(deep=True)
    
    # Drop the columns that we do not need.
    df = df.drop(columns=["feelings_symptoms", "smell_description", "zipcode"])
    
    # Select only the reports within the range of 3 and 5.
    df = df[(df["smell_value"]>=3)&(df["smell_value"]<=5)]
    
    # Convert the timestamp to datetime.
    df.index = pd.to_datetime(df.index, unit="s", utc=True)

    # Resample the timestamps by hour and sum up all the values.
    # Because we want data from the past, so label need to be "right".
    df = df.resample("60Min", label="right").sum()
    
    # Fill in the missing data with value 0.
    df = df.fillna(0)
    return df


def answer_preprocess_sensor(df_list):
    """
    This function is the answer of task 5.
    """
    # Resample all the data frames.
    df_resample_list = []
    for df in df_list:
        # Convert the timestamp to datetime.
        df.index = pd.to_datetime(df.index, unit="s", utc=True)
        # Resample the timestamps by hour and average all the values.
        df_resample_list.append(df.resample("60Min", label="right").mean())
    
    # Merge all data frames.
    df = df_resample_list.pop(0)
    index_name = df.index.name
    while len(df_resample_list) != 0:
        # We need to use outer merging since we want to preserve data from both data frames.
        df = pd.merge_ordered(df, df_resample_list.pop(0), on=df.index.name, how="outer", fill_method=None)
        # Move the datetime column to index
        df = df.set_index(index_name)

    # Fill in the missing data with value -1.
    df = df.fillna(-1)
    return df


# ## Task 4: Preprocess Smell Data

# In this task, we will preprocess the smell data. First, we need to load the raw smell data.

# In[3]:


smell_raw = pd.read_csv("../../../assets/datasets/smellpgh-v1/smell_raw.csv").set_index("EpochTime")
smell_raw


# Next, we need to resample the smell data so that they can be used for modeling. Our goal is to have a dataframe that looks like the following:

# In[4]:


df_smell = answer_preprocess_smell(smell_raw)
df_smell


# **Your task (which is your assignment) is to write a function to do the following:**
# - First, remove the `feelings_symptoms`, `smell_description`, and `zipcode` columns since we do not need them. (Hint: use the `pandas.DataFrame.drop` function)
# - We only want the reports that indicate bad smell. You need to select only the reports with rating 3, 4, or 5 in the `smell_value` column.
# - Then, we want to know the severity of bad smell within an hour. So you need to resample the data by computing the hourly sum of smell values from the previous hour. (Hint: use the `pandas.DataFrame.resample` function)
# - Finally, fill in the missing data with the value 0. The reason is that missing data means there are no smell reports (provided by citizens) within an hour, so we assume that there is no bad smell within this period of time. Notice that this is an assumption and also a limitation since citizens rarely report good smell.

# In[5]:


def preprocess_smell(df):
    ###################################
    # Fill in your answer here
    return None
    ###################################


# The code below tests if the output of your function matches the expected output.

# In[6]:


check_answer_df(preprocess_smell(smell_raw), df_smell, n=1)


# Now, we can plot the distribution of smell values by using the `pandas.DataFrame.plot` function. From the plot below, we can observe that a lot of the time, the smell values are fairly low. This means that smell events only happen occasionally, and thus our dataset is highly imbalanced.

# In[7]:


fig = df_smell.plot(kind="hist", bins=20, ylim=(0,100), edgecolor="black").set_yticks([0,50,100], labels=["0","50",">100"])


# In[ ]:





# ## Task 5: Preprocess Sensor Data

# In this task, we will process the sensor data from various air quality monitoring stations in Pittsburgh. You can find the list of sensors and their names (which will be in the data frame columns) from [this link](https://github.com/CMU-CREATE-Lab/smell-pittsburgh-prediction/tree/master/dataset/v2.1#description-of-the-air-quality-sensor-data). First, we need to load all the sensor data.

# In[8]:


path = "../../../assets/datasets/smellpgh-v1/esdr_raw"
list_of_files = [f for f in listdir(path) if isfile(join(path, f))]
sensor_raw_list = []
for f in list_of_files:
    sensor_raw_list.append(pd.read_csv(join(path, f)).set_index("EpochTime"))


# Now, the `sensor_raw_list` variable contains all the data frames with sensor values from different air quality monitoring stations. Noted that `sensor_raw_list` is an array of data frames. We can print one of them to take a look, as shown below. 

# In[9]:


sensor_raw_list[0]


# Next, we need to resample and merge all the sensor data frames so that they can be used for modeling. Our goal is to have a dataframe that looks like the following:

# In[10]:


df_sensor = answer_preprocess_sensor(sensor_raw_list)
df_sensor


# **Your task (which is your assignment) is to write a function to do the following:**
# - Sensors can report in various frequencies. So, for each data frame, we need to resample the data by computing the hourly average of sensor measurements from the previous hour. (Hint: use the `pandas.DataFrame.resample` function)
# - Then, merge all the data frames based on their time stamp, which is the `EpochTime` column.
# - Finally, fill in the missing data with the value -1. The reason for not using 0 here is that we want the model to know if the sensor measurement has a value (including zero) or missing.

# In[11]:


def preprocess_sensor(df_list):
    ###################################
    # Fill in your answer here
    return None
    ###################################


# The code below tests if the output of your function matches the expected output.

# In[12]:


check_answer_df(preprocess_sensor(sensor_raw_list), df_sensor, n=1)


# In[ ]:




