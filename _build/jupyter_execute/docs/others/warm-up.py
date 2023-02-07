#!/usr/bin/env python
# coding: utf-8

# # Python Coding Warm-Up

# (Last updated: Feb 7, 2023)
# 
# This notebook is designed to help you warm up python programming, specifically related to coding up data science pipelines.

# In[1]:


import pandas as pd
import numpy as np


# **Do not check the answers in the next cell before practicing the tasks.**

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


def answer_resample_df(df):
    """
    This function is the answer for task 1.
    """
    # Copy to avoid modifying the original dataframe.
    df = df.copy(deep=True)

    # Convert the timestamp to datetime.
    df.index = pd.to_datetime(df.index, unit="s", utc=True)

    # Resample the timestamps by hour and take the average value.
    # Because we want data from the past, so label need to be "right".
    df = df.resample("60Min", label="right").mean()
    return df


def answer_merge_df(df1, df2):
    """
    This function is the answer for task 2.
    """
    # Copy to avoid modifying the original dataframe.
    df1 = df1.copy(deep=True)
    df2 = df2.copy(deep=True)

    # Make sure that the index has the same name.
    df2.index.name = df1.index.name

    # Merge the two data frames based on the index name.
    # We need to use outer merging since we want to preserve data from both data frames.
    df = pd.merge_ordered(df1, df2, on=df1.index.name, how="outer", fill_method=None)

    # Move the datetime column to index
    df = df.set_index(df1.index.name)
    return df


def answer_aggregate_df(df):
    """
    This function is the answer for task 3.
    """
    # Copy to avoid modifying the original dataframe.
    df = df.copy(deep=True)

    # Filter the data
    df = df[(df["v1"]>0)&(df["group"]!="15227")]

    # Aggregate data for each group
    all_groups = []
    for g, df_g in df.groupby("group"):
        # Select only the variable v1.
        df_g = df_g["v1"]
        # Resample data using your code (or the answer) for task 1
        df_g = answer_resample_df(df_g)
        # Set the dataframe's name to the group value
        df_g.name = g
        # Save the group in an array
        all_groups.append(df_g)

    # Merge all groups using your code (or the answer) for task 2
    df = all_groups.pop(0)
    while len(all_groups) != 0:
        df = answer_merge_df(df, all_groups.pop(0))

    # Fill in the missing data with value -1
    df = df.fillna(0)
    return df


def answer_transform_df(df):
    """
    This function is the answer for task 4.
    """
    # Copy to avoid modifying the original dataframe.
    df = df.copy(deep=True)

    # Define the function to process wind speed
    def process_wind_mph(x):
        if pd.isna(x):
            return None
        else:
            return x<5

    # Add the transformed columns.
    df["wind_deg_sine"] = np.sin(np.deg2rad(df["wind_deg"]))
    df["wind_deg_cosine"] = np.cos(np.deg2rad(df["wind_deg"]))
    df["is_calm_wind"] = df["wind_mph"].apply(process_wind_mph)

    # Delete the original columns.
    df = df.drop(["wind_deg"], axis=1)
    df = df.drop(["wind_mph"], axis=1)
    return df


def answer_transform_text_df(df):
    """
    This function is the answer for task 5.
    """
    # Copy to avoid modifying the original dataframe.
    df = df.copy(deep=True)

    # Process the required columns.
    df["CV"] = df["venue"].str.contains("BMVC|WACV|ICCV|CVPR")
    df["ML"] = df["venue"].str.contains("NeurIPS|ICLR")
    df["MM"] = df["venue"].str.contains("MM")
    df["year"] = df["venue"].str.extract(r'([0-9]{4})')

    # Delete the venue columns
    df = df.drop(["venue"], axis=1)
    return df


# ## Task 1: Resample Data

# ### Task Description
# 
# - Given a data frame with timestamps and variables, resample the data by computing hourly average values from the previous hour.
# - For example, at 9:00 time, compute the average value for all data points betwee 8:00 and 9:00 and put the average value to a cell.
# - You should use pandas to operate the data frame (but not converting it to something else like an array).
# 
# ### Other Information
# 
# - The timestamp is represented in seconds in epoch time, which means the number of seconds that have elapsed since January 1st, 1970 (midnight UTC/GMT).
# - This task is very common in dealing with structured data. For example, a sensor may report many readings in a high frequency, but we only want the hourly averaged value.
# - Hint: Use the `pandas.DataFrame.resample` function. Check the documentation by typing `?pd.DataFrame.resample` in a notebook cell and run the cell.

# In[3]:


# Below is an example input.
data_task1_case1 = [
[1477899000,52.6],
[1477902600,48.3],
[1477904000,44.2],
[1477906200,31.1],
[1477911200,42.7]]
df_task1_case1 = pd.DataFrame(data=data_task1_case1, columns=["timestamps","v1"]).set_index("timestamps")
df_task1_case1


# In[4]:


# Below is an example output.
answer_resample_df(df_task1_case1)


# In[5]:


def resample_df(df):
    ###################################
    # Fill in your answer here
    return None
    ###################################


# In[6]:


# Check if test case 1 is passed.
check_answer_df(resample_df(df_task1_case1), answer_resample_df(df_task1_case1), n=1)


# In[7]:


# Below is another test case.
data_task1_case2 = [
[1477891800,51.7],
[1477893600,47.2],
[1477895400,52.7],
[1477899000,52.6],
[1477902600,48.3],
[1477904000,44.2],
[1477906200,31.1],
[1477913400,61.2],
[1477917000,77.4],
[1477920100,65.4],
[1477920600,35.3],
[1477924100,13.6],
[1477925300,23.2],
[1477925800,32.6],
[1477926300,42.3]]
df_task1_case2 = pd.DataFrame(data=data_task1_case2, columns=["timestamps","v1"]).set_index("timestamps")
df_task1_case2


# In[8]:


# Check if test case 2 is passed.
check_answer_df(resample_df(df_task1_case2), answer_resample_df(df_task1_case2), n=2)


# In[9]:


# Below is another test case.
data_task1_case3 = [
[1477886800,185.0,1.1],
[1477887200,223.0,2.2],
[1477891800,343.0,1.56],
[1477899000,359.0,5.97],
[1477902600,5.0,3.21],
[1477906200,41.0,9.05],
[1477906800,26.0,7.2],
[1477907500,34.0,3.2],
[1477909800,25.0,1.0],
[1477913400,9.0,0.45],
[1477917000,263.0,2.2],
[1477920000,222.0,3.4],
[1477920600,84.0,1.33]]
df_task1_case3 = pd.DataFrame(data=data_task1_case3, columns=["timestamps","v2","v3"]).set_index("timestamps")
df_task1_case3


# In[10]:


# Check if test case 3 is passed.
check_answer_df(resample_df(df_task1_case3), answer_resample_df(df_task1_case3), n=3)


# In[ ]:





# ## Task 2: Merge Data Frames

# ### Task Description
# 
# - Given two data frames with different timestamps and variables, merge them into a single data frame.
# - You should use pandas to operate the data frame (but not converting it to something else like an array).
# 
# ### Other Information
# 
# - This task is common in dealing with structured data. For example, there can be sensor measurements from many stations, which are logged at different time points. So, we usually need to merge them into a single data frame.
# - This task is in a part of the pipeline that continues Task 1.
# - Hint: Use the `pandas.merge_ordered` function. Check the documentation by typing `?pd.merge_ordered` in a notebook cell and run the cell.

# In[11]:


# Below is an example of the first input:
data_task2_case1_input1 = [
[1477909800,30.9],
[1477913400,61.2],
[1477917000,77.4],
[1477920600,35.3]]
df_task2_case1_input1 = pd.DataFrame(data=data_task2_case1_input1, columns=["timestamps","v1"]).set_index("timestamps")
df_task2_case1_input1 = answer_resample_df(df_task2_case1_input1)
df_task2_case1_input1


# In[12]:


# Below is an example of the second input:
data_task2_case1_input2 = [
[1477909800,25.0],
[1477913400,9.0],
[1477917000,263.0],
[1477920600,84.0]]
df_task2_case1_input2 = pd.DataFrame(data=data_task2_case1_input2, columns=["timestamps","v2"]).set_index("timestamps")
df_task2_case1_input2 = answer_resample_df(df_task2_case1_input2)
df_task2_case1_input2


# In[13]:


# Below is an example output.
answer_merge_df(df_task2_case1_input1, df_task2_case1_input2)


# In[14]:


def merge_df(df1, df2):
    ###################################
    # Fill in your answer here
    return None
    ###################################


# In[15]:


# Check if test case 1 is passed.
check_answer_df(merge_df(df_task2_case1_input1, df_task2_case1_input2), answer_merge_df(df_task2_case1_input1, df_task2_case1_input2), n=1)


# In[16]:


# Below is the first input for another test case.
df_task2_case2_input1 = answer_resample_df(df_task1_case2)
df_task2_case2_input1


# In[17]:


# Below is the second input for another test case.
df_task2_case2_input2 = answer_resample_df(df_task1_case3)
df_task2_case2_input2


# In[18]:


# Check if test case 2 is passed.
check_answer_df(merge_df(df_task2_case2_input1, df_task2_case2_input2), answer_merge_df(df_task2_case2_input1, df_task2_case2_input2), n=2)


# In[ ]:





# ## Task 3: Filter and Aggregate Data

# ### Task Description
# 
# - Given a data frame with variables, follow the steps below:
#   - First, remove the rows with zero and negative values in the "v1" column.
#   - Also, remove the rows with value "15227" in the "group" column. 
#   - Next, group the data points by the "group" column.
#   - Then, for each group, resample the data based on criteria described in task 1 (hourly average).
#   - Finally, fill in the missing data with value 0.
# - Use your code for Task 1 and Task 2 to complete this task faster.
# - You should use pandas to operate the data frame (but not converting it to something else like an array).
# 
# ### Other Information
# 
# - This task is common in dealing with structured data. For example, there can be data from different local regions which are represented by some numbers, such as zip codes. And we only want the data from a certain regions (but not all of them).
# - This task is in a part of the pipeline that continues both Task 1 and Task 2.
# - Hint: Use the `pandas.DataFrame.groupby` function. Check the documentation by typing `?pd.DataFrame.groupby` in a notebook cell and run the cell.

# In[19]:


# Below is an example input.
data_task3_case1 = [
[1477935134,1,"15206"],
[1477935767,1,"15227"],
[1477955141,1,"15207"],
[1477956180,2,"15206"],
[1477956293,-4,"15218"],
[1477973293,5,"15207"]]
df_task3_case1 = pd.DataFrame(data=data_task3_case1, columns=["timestamps","v1","group"]).set_index("timestamps")
df_task3_case1


# In[20]:


# Below is an example output.
answer_aggregate_df(df_task3_case1)


# In[21]:


def aggregate_df(df):
    ###################################
    # Fill in your answer here
    return None
    ###################################


# In[22]:


# Check if test case 1 is passed.
check_answer_df(aggregate_df(df_task3_case1), answer_aggregate_df(df_task3_case1), n=1)


# In[23]:


# Below is another test case.
data_task3_case2 = [
[1477935134,1,"15206"],
[1477935767,1,"15227"],
[1477955141,1,"15207"],
[1477956180,2,"15206"],
[1477956293,-4,"15207"],
[1477970157,3,"15227"],
[1477973293,5,"15207"],
[1478001707,4,"15206"],
[1478001989,2,"15206"],
[1478003840,3,"15206"],
[1478005371,3,"15206"],
[1478005897,3,"15207"],
[1478006679,1,"15206"],
[1478014118,4,"15206"],
[1478014136,-2,"15206"],
[1478014162,4,"15206"],
[1478014317,4,"15207"],
[1478015537,-1,"15206"],
[1478015548,3,"15206"],
[1478015587,3,"15206"]]
df_task3_case2 = pd.DataFrame(data=data_task3_case2, columns=["timestamps","v1","group"]).set_index("timestamps")
df_task3_case2


# In[24]:


# Check if test case 2 is passed.
check_answer_df(aggregate_df(df_task3_case2), answer_aggregate_df(df_task3_case2), n=2)


# In[ ]:





# ## Task 4: Transform Structured Data

# ### Task Description
# 
# - Given a data frame with timestamps and wind information, transform the wind direction to sine/cosine components and threshold the wind speed. 
# - Remove the orignal wind direction column (`wind_deg`) and add two new columns (`wind_deg_sine` and `wind_deg_cosine`) to the data frame.
# - Remove the orignal wind speed column (`wind_mph`) and add a new column (`is_calm_wind`) to show if wind speed is lower than 5 MPH.
# - The `is_calm_wind` value should be `True` (when `wind_mph>=5`), `False` (when `wind_mph<5`), or `None` (when `wind_mph` data is missing).
# - The columns need to follow the order `["wind_deg_sine", "wind_deg_cosine", "is_calm_wind"]`.
# 
# ### Other Information
# 
# - The task is commonly used when encoding cyclical features, such as wind direction (between 0 and 359), time of day (between 0 and 23), and month (between 1 and 12).
# - Hint: Use the `pandas.DataFrame.apply` function. Check the documentation by typing `?pd.DataFrame.apply` in a notebook cell and run the cell.
# - Hint: Use the `pandas.isna` function. Check the documentation by typing `?pd.isna` in a notebook cell and run the cell.
# - Hint: Use the `numpy.sin` and `numpy.cos` function to perform the transformation. Be careful that these functions operate on radians but not degrees.

# In[25]:


# Below is an example input.
data_task4_case1 = [
[1477891800,343.0,3.6],
[1477895400,351.0,None],
[1477899000,359.0,6.4],
[1477902600,5.0,5.1]]
df_task4_case1 = pd.DataFrame(data=data_task4_case1, columns=["timestamps","wind_deg","wind_mph"]).set_index("timestamps")
df_task4_case1


# In[26]:


# Below is an example output.
answer_transform_df(df_task4_case1)


# In[27]:


def transform_df(df):
    ###################################
    # Fill in your answer here
    return None
    ###################################


# In[28]:


# Check if test case 1 is passed.
check_answer_df(transform_df(df_task4_case1), answer_transform_df(df_task4_case1), n=1)


# In[29]:


# Below is another test case.
data_task4_case2 = [
[1477891800,343.0,3.6],
[1477895400,351.0,None],
[1477899000,359.0,6.4],
[1477902600,None,5.1],
[1477906200,41.0,5.2],
[1477909800,25.0,4.6],
[1477913400,9.0,7.7],
[1477917000,263.0,8.8],
[1477920600,84.0,7.9],
[1477924200,None,None],
[1477927800,None,None],
[1477931400,115.0,5.2],
[1477935000,117.0,3.8],
[1477938600,97.0,4.1],
[1477942200,141.0,3.1]]
df_task4_case2 = pd.DataFrame(data=data_task4_case2, columns=["timestamps","wind_deg","wind_mph"]).set_index("timestamps")
df_task4_case2


# In[30]:


# Check if test case 2 is passed.
check_answer_df(transform_df(df_task4_case2), answer_transform_df(df_task4_case2), n=2)


# In[ ]:





# ## Task 5: Transform Text Data

# ### Task Description
# 
# - Given a data frame with information about paper publications, extract the following information:
#   - Add a new column `CV` to indicate if the paper comes from Computer Vision venues (if `venue` column contains string "BMVC", "WACV", "ICCV", or "CVPR").
#   - Add a new column `ML` to indicate if the paper comes from Machine Learning venues (if `venue` column contains string "NeurIPS" or "ICLR").
#   - Add a new column `MM` to indicate if the paper comes from Multimedia venues (if `venue` column contains string "MM").
#   - Add a new column `year` to show the year of publication in the venue string. If year is unknown, put value NaN in the cell.
# - The data type for column `CV`, `ML`, and `MM` should be binary (True/False).
# - The data type for column `year` should be integer.
# - The order of the column needs to be `["CV","ML","MM","year"]`.
# 
# ### Other Information
# 
# - Extracting information from strings is a common task in text processing, such as using these extracted information to classify documents.
# - Hint: Use the `pandas.Series.str.contains` funtcion. Check the documentation by typing `?pd.Series.str.contains` in a notebook cell and run the cell.
# - Hint: Use the `pandas.Series.str.extract` function. Check the documentation by typing `?pd.Series.str.extract` in a notebook cell and run the cell.

# In[31]:


# Below is an example input.
data_task5_case1 = [
[1, "WACV_2023"],
[2, "WACV"],
[3, "2023NeurIPS"],
[4, "Journal of Forensic Sciences 2022"],
[5, "CVPR2022"]]
data_task5_case1 = pd.DataFrame(data=data_task5_case1, columns=["paper_id","venue"]).set_index("paper_id")
data_task5_case1


# In[32]:


# Below is an example output.
answer_transform_text_df(data_task5_case1)


# In[33]:


def transform_text_df(df):
    ###################################
    # Fill in your answer here
    return None
    ###################################


# In[34]:


# Check if test case 1 is passed.
check_answer_df(transform_text_df(data_task5_case1), answer_transform_text_df(data_task5_case1), n=1)


# In[35]:


# Below is another test case.
data_task5_case2 = [
[1, "WACV_2023"],
[2, "WACV"],
[3, "2023NeurIPS"],
[4, "Journal of Forensic Sciences 2022"],
[5, "CVPR2022"],
[6, "Forensic Science International"],
[7, "MM2021"],
[8, "CVPR"],
[9, "ICLR-2021"],
[10, "BVMC 2021"],
[11, "WACV2021"],
[12, "NeurIPS 2021"],
[13, "BMVC"],
[14, "2020 Journal of Interactive Marketing"],
[15, "ACM MM"],
[16, "ICCV2019"],
[17, "ICCV"],
[18, "2019-ICCV"],
[19, "2019MM"],
[20, "WACV2018"],
[21, "2017BMVC"],
[22, "ICLR"],
[23, "Journal of Interactive Marketing 2016"],
[24, "BVMC 2015"],
[25, "2014 NeurIPS"],
[26, "2013 MM"]]
data_task5_case2 = pd.DataFrame(data=data_task5_case2, columns=["paper_id","venue"]).set_index("paper_id")
data_task5_case2


# In[36]:


# Check if test case 2 is passed.
check_answer_df(transform_text_df(data_task5_case2), answer_transform_text_df(data_task5_case2), n=2)

