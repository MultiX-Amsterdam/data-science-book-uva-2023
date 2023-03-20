#!/usr/bin/env python
# coding: utf-8

# # Pytorch Implementation of Smell Prediction

# In[1]:


import pandas as pd
import numpy as np
from os.path import isfile, join
from os import listdir
from copy import deepcopy
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Below we hide a bunch of functions for preprocessing the data.

# In[2]:


def answer_preprocess_sensor(df_list):
    """
    This function is the answer of task 5.
    Preprocess sensor data.
    
    Parameters
    ----------
    df_list : list of pandas.DataFrame
        A list of data frames that contain sensor data from multiple stations.
         
    Returns
    -------
    pandas.DataFrame
        The preprocessed sensor data.
    """
    # Resample all the data frames.
    df_resample_list = []
    for df in df_list:
        # Convert the timestamp to datetime.
        df.index = pd.to_datetime(df.index, unit="s", utc=True)
        # Resample the timestamps by hour and average all the previous values.
        # Because we want data from the past, so label need to be "right".
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


def answer_preprocess_smell(df):
    """
    This function is the answer of task 4.
    Preprocess smell data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The raw smell reports data.
         
    Returns
    -------
    pandas.DataFrame
        The preprocessed smell data.
    """
    # Copy the dataframe to avoid editing the original one.
    df = df.copy(deep=True)
    
    # Drop the columns that we do not need.
    df = df.drop(columns=["feelings_symptoms", "smell_description", "zipcode"])
    
    # Select only the reports within the range of 3 and 5.
    df = df[(df["smell_value"]>=3)&(df["smell_value"]<=5)]
    
    # Convert the timestamp to datetime.
    df.index = pd.to_datetime(df.index, unit="s", utc=True)

    # Resample the timestamps by hour and sum up all the future values.
    # Because we want data from the future, so label need to be "left".
    df = df.resample("60Min", label="left").sum()
    
    # Fill in the missing data with value 0.
    df = df.fillna(0)
    return df


def answer_sum_current_and_future_data(df, n_hr=0):
    """
    This function is the answer of task 6.
    Sum up data in the current and future hours.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The preprocessed smell data.
    n_hr : int
         Number of hours that we want to sum up the future smell data.
         
    Returns
    -------
    pandas.DataFrame
        The transformed smell data.
    """
    # Copy data frame to prevent editing the original one.
    df = df.copy(deep=True)
    
    # Fast return if n_hr is 0
    if n_hr == 0: return df
    
    # Sum up all smell_values in future hours.
    # The rolling function only works for summing up previous values.
    # So we need to shift back to get the value in the future.
    # Be careful that we need to add 1 to the rolling window size.
    # Becasue window size 1 means only using the current data.
    # Parameter "closed" need to be "right" because we want the current data.
    df = df.rolling(n_hr+1, min_periods=1, closed="right").sum().shift(-1*n_hr)
    
    # Delete the last n_hr rows.
    # These n_hr rows have wrong data due to data shifting.
    df = df.iloc[:-1*n_hr]
    return df


def insert_previous_data_to_cols(df, n_hr=0):
    """
    Insert columns to indicate the data from the previous hours.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The preprocessed sensor data.
    n_hr : int
        Number of hours that we want to insert the previous sensor data.
         
    Returns
    -------
    pandas.DataFrame
        The transformed sensor data.
    """
    # Copy data frame to prevent editing the original one.
    df = df.copy(deep=True)

    # Add the data from the previous hours.
    df_all = []
    for h in range(1, n_hr + 1):
        # Shift the data frame to get previous data.
        df_pre = df.shift(h)
        # Edit the name to indicate it is previous data.
        # The orginal data frame already has data from the previous 1 hour.
        # (as indicated in the preprocessing phase of sensor data)
        # So we need to add 1 here.
        df_pre.columns += "_pre_" + str(h+1) + "h"
        # Add the data to an array for merging.
        df_all.append(df_pre)

    # Rename the columns in the original data frame.
    # The orginal data frame already has data from the previous 1 hour.
    # (as indicated in the preprocessing phase of sensor data)
    df.columns += "_pre_1h"

    # Merge all data.
    df_merge = df
    for d in df_all:
        # The join function merges dataframes by index.
        df_merge = df_merge.join(d)
        
    # Delete the first n_hr rows.
    # These n_hr rows have no data due to data shifting.
    df_merge = df_merge.iloc[n_hr:]
    return df_merge


def convert_wind_direction(df):
    """
    Convert wind directions to sine and cosine components.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The data frame that contains the wind direction data.
         
    Returns
    -------
    pandas.DataFrame
        The transformed data frame.
    """
    # Copy data frame to prevent editing the original one.
    df_cp = df.copy(deep=True)
    
    # Convert columns with wind directions.
    for c in df.columns:
        if "SONICWD_DEG" in c:
            df_c = df[c]
            df_c_cos = np.cos(np.deg2rad(df_c))
            df_c_sin = np.sin(np.deg2rad(df_c))
            df_c_cos.name += "_cosine"
            df_c_sin.name += "_sine"
            df_cp.drop([c], axis=1, inplace=True)
            df_cp[df_c_cos.name] = df_c_cos
            df_cp[df_c_sin.name] = df_c_sin
    return df_cp


def compute_feature_label(df_smell, df_sensor, b_hr_sensor=0, f_hr_smell=0):
    """
    Compute features and labels from the smell and sensor data.
    
    Parameters
    ----------
    df_smell : pandas.DataFrame
        The preprocessed smell data.
    df_sensor : pandas.DataFrame
        The preprocessed sensor data.
    b_hr_sensor : int
        Number of hours that we want to insert the previous sensor data.
    f_hr_smell : int
        Number of hours that we want to sum up the future smell data.
    
    Returns
    -------
    df_x : pandas.DataFrame
        The features that we want to use for modeling.
    df_y : pandas.DataFrame
        The labels that we want to use for modeling.
    """
    # Copy data frames to prevent editing the original ones.
    df_smell = df_smell.copy(deep=True)
    df_sensor = df_sensor.copy(deep=True)
    
    # Replace -1 values in sensor data to NaN
    df_sensor[df_sensor==-1] = np.nan
    
    # Convert all wind directions.
    df_sensor = convert_wind_direction(df_sensor)
    
    # Scale sensor data and fill in missing values
    df_sensor = (df_sensor - df_sensor.mean()) / df_sensor.std()
    df_sensor = df_sensor.round(6)
    df_sensor = df_sensor.fillna(-1)
    
    # Insert previous sensor data as features.
    # Noice that the df_sensor is already using the previous data.
    # So b_hr_sensor=0 means using data from the previous 1 hour.
    # And b_hr_sensor=n means using data from the previous n+1 hours.
    df_sensor = insert_previous_data_to_cols(df_sensor, b_hr_sensor)
    
    # Sum up current and future smell values as label.
    # Notice that the df_smell is already the data from the future 1 hour.
    # (as indicated in the preprocessing phase of smell data)
    # So f_hr_smell=0 means using data from the future 1 hour.
    # And f_hr_smell=n means using data from the future n+1 hours.
    df_smell = answer_sum_current_and_future_data(df_smell, f_hr_smell)
    
    # Add suffix to the column name of the smell data to prevent confusion.
    # See the description above for the reason of adding 1 to the f_hr_smell.
    df_smell.columns += "_future_" + str(f_hr_smell+1) + "h"
    
    # We need to first merge these two timestamps based on the available data.
    # In this way, we synchronize the time stamps in the sensor and smell data.
    # This also means that the sensor and smell data have the same number of data points.
    df = pd.merge_ordered(df_sensor.reset_index(), df_smell.reset_index(), on=df_smell.index.name, how="inner", fill_method=None)
    
    # Sanity check: there should be no missing data.
    assert df.isna().sum().sum() == 0, "Error! There is missing data."
    
    # Separate features (x) and labels (y).
    df_x = df[df_sensor.columns]
    df_y = df[df_smell.columns]
    
    # Add the hour of day and the day of week.
    dow_radian = df["EpochTime"].dt.dayofweek.copy(deep=True) * 2 * np.pi / 6.0
    tod_radian = df["EpochTime"].dt.hour.copy(deep=True) * 2 * np.pi / 23.0
    df_x.loc[:,"day_of_week_sine"] = np.sin(dow_radian)
    df_x.loc[:,"day_of_week_cosine"] = np.cos(dow_radian)
    df_x.loc[:,"hour_of_day_sine"] = np.sin(tod_radian)
    df_x.loc[:,"hour_of_day_cosine"] = np.cos(tod_radian)
    return df_x, df_y


# In[3]:


# Load and preprocess sensor data
path = "smellpgh-v1/esdr_raw"
list_of_files = [f for f in listdir(path) if isfile(join(path, f))]
sensor_raw_list = []
for f in list_of_files:
    sensor_raw_list.append(pd.read_csv(join(path, f)).set_index("EpochTime"))
df_sensor = answer_preprocess_sensor(sensor_raw_list)

# Load and preprocess smell data
smell_raw = pd.read_csv("smellpgh-v1/smell_raw.csv").set_index("EpochTime")
df_smell = answer_preprocess_smell(smell_raw)

# Compute features and labels
df_x, df_y = compute_feature_label(df_smell, df_sensor, b_hr_sensor=2, f_hr_smell=7)


# In[4]:


df_x


# In[5]:


df_y


# In[6]:


# Set random seed for reproducibility
torch.manual_seed(42)

# Load data
feature = df_x[df_x.columns].to_numpy()
label = (df_y>=40).astype(int)['smell_value_future_8h'].to_numpy()

# Create the dataset object
class SmellPittsburghDataset(Dataset):
    def __init__(self, feature=None, label=None):
        self.feature = feature
        self.label = label

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        x = self.feature[idx]
        y = self.label[idx]
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(np.array([y])).float()
        return x, y


# In[7]:


def scorer(y_predict, y):
    """
    A customized scoring function to evaluate a PyTorch classifier.
    
    Parameters
    ----------
    y_predict : torch.Tensor
        The predicted labels.
    y : torch.Tensor
        The true labels.
    
    Returns
    -------
    dict of int or float
        A dictionary of evaluation metrics.
    """
    c = confusion_matrix(y, y_predict, labels=[0,1])
    return {"tn": c[0,0], "fp": c[0,1], "fn": c[1,0], "tp": c[1,1]}


# In[8]:


def train(model, criterion, optimizer, dataloader_train, dataloader_test, num_epochs=30):
    """Train the model."""
    
    def run_one_epoch(dataloader, phase="train"):
        if phase == "train": model.train() # training mode
        else: model.eval() # evaluation mode
        c = 0 # just a counter
        accu_loss = 0 # accumulated loss
        accu_score = None # accumulated scores
        # Loop the data
        for x, y in dataloader:
            c += 1 # increase the counter
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if phase == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Store statistics for the training set
            accu_loss += loss # add up the loss
            y_label = (y_pred > 0.5).float()
            score = scorer(y_label, y)
            if accu_score is None:
                accu_score = score
            else:
                for k in score:
                    accu_score[k] += score[k]
        # Return statistics
        return accu_loss/c, accu_score
    
    def compute_statistics(score):
        tp_fp = score["tp"] + score["fp"]
        if tp_fp == 0:
            precision = 0
        else:
            precision = round(score["tp"]/tp_fp, 2)
        tp_fn = score["tp"] + score["fn"]
        if tp_fn == 0:
            recall = 0
        else:
            recall = round(score["tp"]/tp_fn, 2)
        tp_tp_fp_fn = tp_fp + tp_fn
        if tp_tp_fp_fn == 0:
            f1 = 0
        else:
            f1 = round(2*score["tp"]/tp_tp_fp_fn, 2)
        return precision, recall, f1
    
    # Run one epoch
    for epoch in range(num_epochs):
        # Run through the entire training set
        loss_train, score_train = run_one_epoch(dataloader_train, phase="train")
        loss_train = torch.round(loss_train, decimals=2)
        p_train, r_train, f1_train = compute_statistics(score_train)
        # Run through the entire testing set
        with torch.no_grad():
            loss_test, score_test = run_one_epoch(dataloader_test, phase="test")
        loss_test = torch.round(loss_test, decimals=2)
        p_test, r_test, f1_test = compute_statistics(score_test)
        # Print loss and scores
        if ((epoch+1)%30 == 0):
            print(f"-"*10)
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"Training loss: {loss_train:.4f}, prevision: {p_train:.2f}, recall: {r_train:.2f}, f1: {f1_train:.2f}")
            print(f"Training evaluation: {score_train}")
            print(f"Testing loss: {loss_test:.4f}, prevision: {p_test:.2f}, recall: {r_test:.2f}, f1: {f1_test:.2f}")
            print(f"Testing evaluation: {score_test}")
    
    # Return statistics
    return p_test, r_test, f1_test


# In[9]:


# Define neural network model
class DeepLogisticRegression(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(DeepLogisticRegression, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


# In[10]:


# Create time series splits for cross-validation.
splits = []
dataset_size = df_x.shape[0]
train_size = 8000
test_size = 168
input_size = feature.shape[1]
for i in range(train_size, dataset_size, test_size):
    start = i - train_size
    end = i + test_size
    if (end >= dataset_size): break
    train_index = range(start, i)
    test_index = range(i, end)
    splits.append((list(train_index), list(test_index)))
    
# Cross-validate the model for every split
precision_list = []
recall_list = []
f1_list = []
for i in range(len(splits)):
    print(f"Split: {i}")
    dataset_train = SmellPittsburghDataset(feature=feature[splits[i][0]], label=label[splits[i][0]])
    dataset_test = SmellPittsburghDataset(feature=feature[splits[i][1]], label=label[splits[i][1]])
    dataloader_train = DataLoader(dataset_train, batch_size=1024, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=1024, shuffle=False)
    model = DeepLogisticRegression(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    p_test, r_test, f1_test = train(model, criterion, optimizer, dataloader_train, dataloader_test)
    precision_list.append(p_test)
    recall_list.append(r_test)
    f1_list.append(f1_test)
    print("="*30)


# In[11]:


# Print the overall performance
print("average precision:", round(np.mean(precision_list), 2))
print("average recall:", round(np.mean(recall_list), 2))
print("average f1-score:", round(np.mean(f1_list), 2))


# In[ ]:




