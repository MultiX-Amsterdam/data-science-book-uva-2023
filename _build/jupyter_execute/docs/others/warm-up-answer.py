#!/usr/bin/env python
# coding: utf-8

# # Answer: Python Coding Warm-Up

# (Last updated: Feb 3, 2023)
# 
# This notebook contains the answer for the [python coding warm-up file](warm-up.ipynb).

# In[1]:


import pandas as pd
import numpy as np


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


# **Do not check the answers below before practicing the tasks.**

# In[3]:


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


# In[4]:


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


# In[5]:


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


# In[6]:


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


# In[7]:


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

