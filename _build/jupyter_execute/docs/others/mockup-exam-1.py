#!/usr/bin/env python
# coding: utf-8

# # Mockup Exam 1

# In[1]:


import pandas as pd


# ## Question 1: One-Hot Encoding

# Given a webpage with unknown number of images and a list of images selected by the users, perform one-hot encoding on the selected image IDs.

# In[2]:


# Below is an example input.
data_q1_case1 = [
[1, ["1","2","10"]],
[2, ["1"]],
[3, []],
[4, ["4","2","11","7","5"]],
[5, ["4","5","7","1"]],
[6, ["1","4"]],
[7, ["6","8"]],
[8, ["9","4","5"]]]
data_q1_case1 = pd.DataFrame(data=data_q1_case1, columns=["user_id","image_id_list"]).set_index("user_id")
data_q1_case1


# In[ ]:




