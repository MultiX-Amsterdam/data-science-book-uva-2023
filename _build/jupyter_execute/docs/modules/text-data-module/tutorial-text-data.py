#!/usr/bin/env python
# coding: utf-8

# # Tutorial (Text Data Processing)

# (Last updated: Mar 6, 2023)

# This tutorial will familiarize you with the data science pipeline of processing text data. We will go through the various steps involved in the Natural Language Processing (NLP) pipeline for topic modelling and topic classification, including tokenization, lemmatization, and obtaining word embeddings. We will also build a neural network using PyTorch for multi-class topic classification using the dataset.
# The AG's News Topic Classification Dataset contains news articles from four different categories, making it a nice source of text data for NLP tasks. We will guide you through the process of understanding the dataset, implementing various NLP techniques, and building a model for classification.

# You can use the following links to jump to the tasks and assignments:
# 
# [table of contents]

# ## Scenario

# The [AG's News Topic Classification Dataset](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv) is a collection of over 1 million news articles from more than 2000 news sources. The dataset was created by selecting the 4 largest classes from the original corpus, resulting in 120,000 training samples and 7,600 testing samples. The dataset is provided by the academic community for research purposes in data mining, information retrieval, and other non-commercial activities. We will use it to demonstrate various NLP techniques on real data, and in the end make 2 models with this data. The files train.csv and test.csv contain all the training and testing samples as comma-separated values with 3 columns: class index, title, and description. Download train.csv and test.csv for the following tasks. 

# ## Import Packages

# We put all the packages that are needed for this tutorial below:

# In[1]:


import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import torch
import torch.nn as nn
import torch.optim as optim

from gensim.models import Word2Vec

from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, confusion_matrix

from tqdm.notebook import tqdm

from xml.sax import saxutils as su


# ## Task Answers

# The code block below contains answers for the assignments in this tutorial. **Do not check the answers in the next cell before practicing the tasks.**

# In[81]:


def check_answer_df(df_result, df_answer, n=1):
    """
    This function checks if two output dataframes are the same.
    
    Parameters
    ----------
    df_result : pandas.DataFrame
        The result from the output of a function.
    df_answer: pandas.DataFrame
        The expected output of the function.
    n : int
        The numbering of the test case.
    """
    try:
        if df_answer.isinstance(list):
            assert any([answer.equals(df_result) for answer in df_answer])
        else:
            assert df_answer.equals(df_result)
        print(f"Test case {n} passed.")
    except:
        print(f"Test case {n} failed.")
        print("")
        print("Your output is:")
        print(df_result)
        print("")
        print("Expected output is", end="")
        if df_answer.isinstance(list):
            print(" one of", end="")
        print(":")
        print(df_answer)
        
def answer_tokenize_and_lemmatize(df):
    """
    Tokenize and lemmatize the text in the dataset.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing at least the text column.
         
    Returns
    -------
    pandas.DataFrame
        The dataframe with the added tokens column.
    """
    # Copy the dataframe to avoid editing the original one.
    df = df.copy(deep=True)

    # Apply the tokenizer to create the tokens column.
    df['tokens'] = df['text'].apply(word_tokenize)
    
    # Apply the lemmatizer on every word in the tokens list.
    df['tokens'] = df['tokens'].apply(lambda tokens: [lemmatizer.lemmatize(token, wordnet_pos(tag)) for token, tag in nltk.pos_tag(tokens)])
    return df


def answer_most_used_words(df, token_col='tokens'):
    """
    Generate a dataframe with the 5 most used words per class, and their count.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing at least the class and tokens columns.
         
    Returns
    -------
    pandas.DataFrame
        The dataframe with 5 rows per class, and an added 'count' column.
        The dataframe is sorted in ascending order on the class and in descending order on the count.
    """
    # Copy the dataframe to avoid editing the original one.
    df = df.copy(deep=True)
    
    # Filter out non-words
    df[token_col] = df[token_col].apply(lambda tokens: [token for token in tokens if token.isalpha()])
    
    # Explode the tokens so that every token gets its own row.
    df = df.explode(token_col)
    
    # Option 1: groupby on class and token, get the size of how many rows per item, 
    # add that as a column.
    counts = df.groupby(['class', token_col]).size().reset_index(name='count')
    
    # Option 2: make a pivot table based on the class and token based on how many
    # rows per combination there are , add counts as a column.
    # counts = counts.pivot_table(index=['class', 'tokens'], aggfunc='size').reset_index(name='count')
    
    # Sort the values on the class and count, get only the first 5 rows per class.
    counts = counts.sort_values(['class', 'count'], ascending=[True, False]).groupby('class').head()

    return counts

def answer_remove_stopwords(df):
    """
    Remove stopwords from the tokens.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing at least the tokens column.
         
    Returns
    -------
    pandas.DataFrame
        The dataframe with stopwords removed from the tokens column.
    """
    # Copy the dataframe to avoid editing the original one.
    df = df.copy(deep=True)
    
    # Using a set for quicker lookups.
    stopwords_set = set(stopwords_list)
    
    # Filter stopwords from tokens.
    df['tokens'] = df['tokens'].apply(lambda tokens: [token for token in tokens if token.lower() not in stopwords_set])
    
    return df


# ## Task 3: Preprocess Text Data

# In this task, we will preprocess the text data from the AG News Dataset. First, we need to load the files.

# In[3]:


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

display(train_df, test_df)


# As you can see, all the classes are distributed evenly in the train and test data.

# In[4]:


display(train_df['Class Index'].value_counts(), test_df['Class Index'].value_counts())


# To make the data more understandable, we will make the classes more understandable by adding a `class` column from the original `Class Index` column, containing the category of the news article. To process both the title and news text together, we will combine the `Title` and `Description` columns into one `text` column. We will just deal with the train data until the point where we need the test data again.

# In[5]:


def reformat_data(df):
    """
    Reformat the Class Index column to a Class column and combine
    the Title and Description columns into a Text column.
    Select only the class_idx, class and text columns afterwards.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The original dataframe.
         
    Returns
    -------
    pandas.DataFrame
        The reformatted dataframe.
    """
    # Make the class column using a dictionary.
    df = df.rename(columns={"Class Index": "class_idx"})
    classes = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
    df['class'] = df['class_idx'].apply(classes.get)
    
    # Use string concatonation for the Text column and unesacpe html characters.
    df['text'] = (df['Title'] + ' ' + df['Description']).apply(su.unescape)
    
    # Select only the Class and Text columns.
    df = df[['class_idx', 'class', 'text']]
    return df

train_df = reformat_data(train_df)
display(train_df)


# ### Tokenization 

# Tokenization is the process of breaking down a text into individual tokens, which are usually words but can also be phrases or sentences. It helps language models to understand and analyze text data by breaking it down into smaller, more manageable pieces. While it may seem like a trivial task, tokenization can be applied in multiple ways and thus be a complex and challenging task influencing NLP applications.
# 
# For example, in languages like English, it is generally straightforward to identify words by using spaces as delimiters. However, there are exceptions, such as contractions like "can't" and hyphenated words like "self-driving". And in Dutch, where multiple nouns can be combined into one bigger noun without any delimiter this can be hard. How would you tokenize "hippopotomonstrosesquippedaliofobie"? In other languages, such as Chinese and Japanese, there are no spaces between words, so identifying word boundaries is much more difficult. 
# 
# To illustrate the use of tokenization, let's consider the following example, which tokenizes a sample text using the `word_tokenize` function from the NLTK package. That function uses a pre-trained tokenization model for English.

# In[6]:


# Sample text.
text = "The quick brown fox jumped over the lazy dog. The cats couldn't wait to sleep all day."

# Tokenize the text.
tokens = word_tokenize(text)

# Print the text and the tokens.
print("Original text:", text)
print("Tokenized text:", tokens)


# ### Part-of-speech tagging

# Part-of-speech (POS) tagging is the process of assigning each word in a text corpus with a specific part-of-speech tag based on its context and definition. The tags typically include nouns, verbs, adjectives, adverbs, pronouns, preposition, conjunction, interjection, and more. POS tagging can help other NLP tasks disambiguate a token in some way due to the added context.

# In[73]:


pos_tags = nltk.pos_tag(tokens)
print(pos_tags)


# ### Stemming / lemmatization

# Stemming and lemmatization are two common techniques used in NLP to preprocess and normalize text data. Both techniques involve transforming words into their root form, but they differ in their approach and the level of normalization they provide.
# 
# Stemming is a technique that involves reducing words to their base or stem form by removing any affixes or suffixes. For example, the stem of the word "lazily" would be "lazi". Stemming is a simple and fast technique that can be useful. However, it can also produce inaccurate or incorrect results since it does not consider the context or part of speech of the word.
# 
# Lemmatization, on the other hand, is a more sophisticated technique that involves identifying the base or dictionary form of a word, also known as the lemma. Unlike stemming, lemmatization can consider the context and part of speech of the word, which can make it more accurate and reliable. With lemmatization, the lemma of the word "lazily" would be "lazy". Lemmatization can be slower and more complex than stemming but provides a higher level of normalization.

# In[76]:


# Initialize the stemmer and lemmatizer.
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

def wordnet_pos(nltk_pos):
    """
    Function to map POS tags to wordnet tags for lemmatizer
    """
    if nltk_pos.startswith('V'):
        return wordnet.VERB
    elif nltk_pos.startswith('J'):
        return wordnet.ADJ
    elif nltk_pos.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

# Perform stemming and lemmatization seperately on the tokens.
stemmed_tokens = [stemmer.stem(token) for token in tokens]
lemmatized_tokens = [lemmatizer.lemmatize(token, wordnet_pos(tag)) for token, tag in nltk.pos_tag(tokens)]

# Print the results.
print("Stemmed text:", stemmed_tokens)
print("Lemmatized text:", lemmatized_tokens)


# ### Stopword removal

# Stopword removal is a common technique used in NLP to preprocess and clean text data by removing words that are considered to be of little or no value in terms of conveying meaning or information. These words are called "stopwords" and they include common words such as "the", "a", "an", "and", "or", "but", and so on.
# 
# The purpose of stopword removal in NLP is to improve the accuracy and efficiency of text analysis and processing by reducing the noise and complexity of the data. Stopwords are often used to form grammatical structures in a sentence, but they do not carry much meaning or relevance to the main topic or theme of the text. So by removing these words, we can reduce the dimensionality of the text data, improve the performance of machine learning models, and speed up the processing of text data. NLTK has a predefined list of stopwords for English.

# In[77]:


# English stopwords in NLTK.
stopwords_list = stopwords.words('english')
print(stopwords_list)


# ### Assignment for Task 3

# **Your task (which is your assignment) is to write functions to do the following:**
# - Since we want to use our text to make a model later on, we need to preprocess it. Add a `tokens` column to the `train_df` dataframe with the text tokenized, then lemmatize those tokens. You must use the POS tags when lemmatizing.
#     - Hint: Use the `pandas.Series.apply` function with the imported `nltk.tokenize.word_tokenize` function. This might take a moment. Recall that you can use the `pd.Series.apply?` syntax in a code cell for more information.
#     - Hint: use the `nltk.stem.WordNetLemmatizer.lemmatize` function to lemmatize a token.
# 
#     Our goal is to have a dataframe that looks like the following:

# In[85]:


display(answer_tokenize_and_lemmatize(train_df))


# - To see what the most used words per class are, create a new, seperate dataframe with the 5 most used words per class. Sort the resulting dataframe ascending on the `class` and descending on the `count`.
#     - Hint: use the `pandas.Series.apply` and `str.isalpha()` functions to filter out non-alphabetical tokens.
#     - Hint: use the `pandas.DataFrame.explode` to create one row per class and token.
#     - Hint: use `pandas.DataFrame.groupby` with `.size()` afterwards or `pandas.DataFrame.pivot_table` with `size` as the `aggfunc` to obtain the occurences per class.
#     - Hint: use the `pandas.Series.reset_index` function to obtain a dataframe with `[class, tokens, count]` as the columns.
#     - Hint: use the `pandas.DataFrame.sort_values` function for sorting a dataframe.
#     - Hint: use the `pandas.DataFrame.groupby` and `pandas.DataFrame.head` functions to get the first 5 rows per class.
#         
#     Our goal is to have a dataframe that looks like the following:

# In[83]:


display(answer_most_used_words(train_df))


# - Remove the stopwords from the `tokens` column in the `train_df` dataframe. Do the most used tokens say something about the class now?
#     - Hint: once again, you can use the `pandas.Series.apply` function. 
#     
#     The top 5 words per class should look like this after removing stopwords:

# In[84]:


display(answer_most_used_words(answer_remove_stopwords(train_df)))


# In[78]:


def tokenize_and_lemmatize(df):
    """
    Tokenize and lemmatize the text in the dataset.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing at least the text column.
         
    Returns
    -------
    pandas.DataFrame
        The dataframe with the added tokens column.
    """
    ###################################
    # Fill in your answer here
    return None
    ###################################


def most_used_words(df, token_col='tokens'):
    """
    Generate a dataframe with the 5 most used words per class, and their count.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing at least the class and tokens columns.
         
    Returns
    -------
    pandas.DataFrame
        The dataframe with 5 rows per class, and an added 'count' column.
        The dataframe is sorted in ascending order on the class and in descending order on the count.
    """
    ###################################
    # Fill in your answer here
    return None
    ###################################

def remove_stopwords(df):
    """
    Remove stopwords from the tokens.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing at least the tokens column.
         
    Returns
    -------
    pandas.DataFrame
        The dataframe with stopwords removed from the tokens column.
    """
    ###################################
    # Fill in your answer here
    return None
    ###################################


# The code below tests if the all your functions matches the expected output.

# In[ ]:


check_answer_df(most_used_words(remove_stopwords(remove_stopwords)), tokenize_and_lemmatize, n=1)


# ## Task 4: Another option: spaCy

# spaCy is another library used to perform various NLP tasks like tokenization, part-of-speech tagging, named entity recognition, dependency parsing, and much more. It provides pre-trained models for different languages and domains, which can be used as-is but also can be fine-tuned on a specific task or domain.
# 
# In an object-oriented way, spaCy can be thought of as a collection of classes and objects that work together to perform NLP tasks. Some of the important functions and classes in spaCy include:
# 
# - `nlp`: The core function that provides the main functionality of spaCy. It is used to process text and create a `Doc` object.
# - [`Doc`](https://spacy.io/api/doc): A container for accessing linguistic annotations like tokens, part-of-speech tags, named entities, and dependency parse information. It is created by the `nlp` function and represents a processed document.
# - [`Token`](https://spacy.io/api/token): An object representing a single token in a `Doc` object. It contains information like the token text, part-of-speech tag, lemma, embedding, and much more.
# 
# When a text is processed by spaCy, it is first passed to the nlp function, which uses the loaded model to tokenize the text and applies various linguistic annotations like part-of-speech tagging, named entity recognition, and dependency parsing in the background. The resulting annotations are stored in a Doc object, which can be accessed and manipulated using various methods and attributes. For example, the Doc object can be iterated over to access each Token object in the document.

# In[10]:


# Load the small English model in spaCy.
# Disable Named Entity Recognition in the model pipeline since we're not using it.
nlp = spacy.load("en_core_web_sm", disable=['ner'])

# Process the text using spaCy.
doc = nlp(text)

# This becomes a spaCy Doc object, which prints nicely as the original string.
print(type(doc) , doc)

# We can iterate over the tokens in the Doc, since it has already been tokenized underneath.
print(type(doc[0]))
for token in doc:
    print(token)


# Since a lot of processing has already been done, we can also directly access multiple attributes of the `Token` objects. For example, we can directly access the lemma of the token with `Token.lemma_` and check if a token is a stop word with `Token.is_stop`.

# In[11]:


print(doc[0].lemma_, type(doc[0].lemma_), doc[0].is_stop, type(doc[0].is_stop))


# Here is the code to add a column with a `Doc` representation of the `text` column to the dataframe. Executing this cell takes several minutes, so we added a progress bar.

# In[12]:


def add_spacy(df):
    """
    Add a column with the spaCy Doc objects.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing at least the text column.
         
    Returns
    -------
    pandas.DataFrame
        The dataframe with the added doc column.
    """
    # Copy the dataframe to avoid editing the original one.
    df = df.copy(deep=True)
    
    df['doc'] = [doc for doc in tqdm(nlp.pipe(df['text']), total=df.shape[0])]
    
    return df

train_df = add_spacy(train_df)


# ### Assignment for Task 4

# **Your task (which is your assignment) is to write a function to do the following:**
# - Add a `spacy_tokens` column containing the to the `train_df` dataframe containing a list of lemmatized tokens (strings). 

# In[13]:


from tqdm.notebook import tqdm

def spacy_tokens(df):
    """
    Add a column with a list of lemmatized tokens, without stopwords.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing at least the doc column.
         
    Returns
    -------
    pandas.DataFrame
        The dataframe with the spacy_tokens column.
    """
    # Copy the dataframe to avoid editing the original one.
    df = df.copy(deep=True)
    
    df['spacy_tokens'] = df['doc'].apply(lambda tokens: [token.lemma_ for token in tokens if not token.is_stop])
    
    return df

train_df = spacy_tokens(train_df)


# We use the answer version of the `most_used_words` function to again display the top 5 words per class in the dataset. Do you see some differences between the lemmatized tokens obtained from NLTK and spaCy?

# In[14]:


display(most_used_words(train_df, 'spacy_tokens'))


# ## Task 5: Unsupervised Learning - Topic Modelling

# Topic modeling is a technique used in NLP that aims to identify the underlying topics or themes in a collection of texts. One way to perform topic modelling is using the probabilistic model Latent Dirichlet Allocation (LDA).
# 
# LDA assumes that each document in a collection is a mixture of different topics, and each topic is a probability distribution over a set of words. The model then infers the underlying topic distribution for each document in the collection and the word distribution for each topic. LDA is trained using an iterative algorithm that maximizes the likelihood of observing the given documents.
# 
# To use LDA, we need to represent the documents as a bag of words, where the order of the words is ignored and only the frequency of each word in the document is considered. This bag-of-words representation allows us to represent each document as a vector of word frequencies, which can be used as input to the LDA algorithm. The Topic Modelling might take a moment on our dataset size.

# In[15]:


# Define the number of topics to extract with LDA
num_topics = 4

# Convert preprocessed text to bag-of-words representation using CountVectorizer.
vectorizer = CountVectorizer(max_features=50000)

# fit_transform requires a string or multiple extra arguments and functions, so turn tokens into string.
X = vectorizer.fit_transform(train_df['spacy_tokens'].apply(lambda x: ' '.join(x)).values)

# Fit LDA to the feature matrix.
lda = LatentDirichletAllocation(n_components=num_topics, max_iter=10, random_state=42, verbose=True)
lda.fit(X)

# Extract the topic proportions for each document.
doc_topic_proportions = lda.transform(X)


# Using this function, we can look at the most important words per topic. Do you see any similarities with the most occuring words per class after stopword removal?

# In[16]:


def n_top_wordlist(model, features, ntopwords=5):
    """
    Add a column with a list of lemmatized tokens, without stopwords.
    """
    output = {}
    for topic_idx, topic in enumerate(model.components_):
        output[topic_idx] = [features[i] for i in topic.argsort()[:-ntopwords - 1:-1]]
    return output

# Get the words from the CountVectorizer.
tf_feature_names = vectorizer.get_feature_names_out()

display(n_top_wordlist(lda, tf_feature_names))


# ### Evaluation

# Adjusted Mutual Information (AMI) and Adjusted Rand Index (ARI) are two metrics used to evaluate the performance of clustering algorithms.
# 
# AMI is a measure that takes into account the possibility of two random clusters appearing to be similar. It is calculated as the difference between the Mutual Information (MI) of two clusterings and the expected MI, divided by the average entropy of the two clusterings minus the expected MI. AMI ranges between 0 and 1, where 0 indicates no agreement between the two clusterings and 1 indicates identical clusterings.
# 
# The Rand Index (RI) is a measure that counts the number of pairs of samples that are assigned to the same or different clusters in both the predicted and true clusterings. The raw RI score is then adjusted for chance into the ARI score using a scheme similar to that of AMI. For ARI a score of 0 indicates random labeling and 1 indicates perfect agreement. The ARI is bounded below by -0.5 for very large differences in labeling.

# ### Assignment for Task 5

# **Your task (which is your assignment) is to write a function to do the following:**
# - The `doc_topic_proportions` contains the proportions of how much that document belongs to every topic. For every document, get the topic in which it has the largest proportion. Afterwards, look at the AMI and ARI scores. Can you improve the scores by modeling more topics or using a different set of tokens?
#     - Hint: use the `numpy.argmax` function.

# In[17]:


def largest_proportion(arr):
    """
    For every row, get the column number where it has the largest value.
    
    Parameters
    ----------
    arr : numpy.array
        The array with the amount of topics as the amount of columns
        and the amount of documents as the number of rows.
        Every row should sum up to 1.
         
    Returns
    -------
    pandas.DataFrame
        The 1-dimensional array containing the label of the topic
        the document has the largest proportion in.
    """
    return np.argmax(arr, axis=1)


# In[67]:


assert True


# In[18]:


topic_most = largest_proportion(doc_topic_proportions)

ami_score = adjusted_mutual_info_score(train_df['class'], topic_most)
ari_score = adjusted_rand_score(train_df['class'], topic_most)

print(f"Adjusted mutual information score: {ami_score:.2f}")
print(f"Adjusted rand score: {ari_score:.2f}")


# ## Task 6: Word embeddings

# Word embeddings represent words as vectors in a high-dimensional space. The key idea behind word embeddings is that words with similar meanings tend to appear in similar contexts, and therefore their vector representations should be close together in this high-dimensional space. Word embeddings have been widely used in various NLP tasks such as sentiment analysis, machine translation, and information retrieval.
# 
# There are several techniques to generate word embeddings, but one of the most popular methods is the Word2Vec algorithm, which is based on a neural network architecture. Word2Vec learns embeddings by predicting the probability of a word given its context (continuous bag of words or skip-gram model). The output of the network is a set of word vectors that can be used as embeddings. Another popular algorithm for generating word embeddings is GloVe (Global Vectors), which is based on matrix factorization techniques.

# In[19]:


# # Load preprocessed text data
# data = pd.read_csv('preprocessed_text.csv')

# # Define the preprocessing functions
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

# def preprocess_text(text):
#     tokens = word_tokenize(text)
#     tokens = [token for token in tokens if token not in stop_words]
#     tokens = [lemmatizer.lemmatize(token) for token in tokens]
#     return tokens

# # Apply the preprocessing function to the text data
# data['tokens'] = data['preprocessed_text'].apply(preprocess_text)

# # Train a Word2Vec model on the preprocessed text data
# model = Word2Vec(data['tokens'], size=100, window=5, min_count=1, workers=4)

# # Get the word embedding for a specific word
# embedding = model.wv['word']


# In[20]:


# import spacy

# # Load the pre-trained spaCy model
# nlp = spacy.load('en_core_web_sm')

# # Load preprocessed text data
# # data = pd.read_csv('preprocessed_text.csv')

# # # Define the preprocessing function
# # def preprocess_text(text):
# #     doc = nlp(text)
# #     tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
# #     return tokens

# # # Apply the preprocessing function to the text data
# # data['tokens'] = data['preprocessed_text'].apply(preprocess_text)

# # Get the word embedding for a specific word
# embedding = nlp('.').vector
# embedding


# ### Assignment for Task 6

# - Sample down the dataset to only use 10% of the original rows.
#     - Hint: use `pandas.DataFrame.sample`.
# - Add a `tensor` column to the dataframe, which is an array containing all the word embedding vectors as columns.
# - Pad all arrays in the `tensor` column to the same number of columns.

# In[ ]:


# Not final yet 

# # Assuming the original dataset is stored in a pandas dataframe called "df"
# # and the word embedding vectors are stored as columns called "emb1", "emb2", ...

# # Sample down to 10% of the original rows
# df = df.sample(frac=0.1, random_state=42)

# # Define a function to convert a row of embeddings to a padded numpy array
# def pad_emb(row, max_len):
#     return np.pad(row.values, (0, max_len - len(row)), mode='constant')

# # Add a "tensor" column to the dataframe, containing the padded embeddings as arrays
# max_len = max(len(row) - 1 for row in df.itertuples())  # Find the maximum number of embeddings
# df['tensor'] = df.apply(lambda row: pad_emb(row[1:-1], max_len), axis=1)

# # Convert the "tensor" column to a 3D numpy array, with shape (num_rows, max_len, num_cols)
# num_cols = len(df.iloc[0]['tensor'])
# tensor = np.stack(df['tensor'].to_numpy(), axis=0).reshape(-1, max_len, num_cols)

# # Example usage: access the first padded tensor in the dataframe
# print(df.iloc[0]['tensor'])  # prints an array of shape (max_len,)
# print(tensor[0])  # prints the same array, but reshaped to (max_len, num_cols)


# ## Task 7: Supervised Learning - Topic Classification

# - Using the word embeddings features, train a small neural net 
# - Don't give the full torch code, only one layer to let them do something with torch
# - Evaluate using confusion matrix against true features
# - Let students be able to tune parameters + n_layers to see if they get better results
# 
# Sources:
# - https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
# - https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

# In[21]:


test_df = spacy_tokens(add_spacy(reformat_data(test_df)))


# In[22]:


most_used_words(test_df, 'spacy_tokens')


# In[ ]:


train_df = train_df.sample(frac=0.1, random_state=42)    
test_df = train_df.sample(frac=0.1, random_state=42)   


# In[66]:


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import pandas as pd
# import numpy as np
# from torch.nn.utils.rnn import pad_sequence


# # Define the neural network model
# class Net(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = nn.functional.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # Define the dataset
# class MyDataset(Dataset):
#     def __init__(self, data):
#         self.data = data

#     def __getitem__(self, index):
#         x = self.data.iloc[index]['tensor']
#         y = self.data.iloc[index]['class_idx']
#         return x, y

#     def __len__(self):
#         return len(self.data) 

# # Combine train and test sets
# combined_df = pd.concat([train_df, test_df], axis=0)

# # Get the maximum sequence length
# max_seq_length = combined_df['tensor'].apply(lambda x: x.shape[1])

# # Pad the sequences to the maximum length
# train_padded = pad_sequence(train_df['tensor'], batch_first=True, padding_value=0, total_length=max_seq_length)
# test_padded = pad_sequence(test_df['tensor'], batch_first=True, padding_value=0, total_length=max_seq_length)

# # Convert labels to tensors
# train_y = torch.tensor(train_df['class_idx'].values)
# test_y = torch.tensor(test_df['class_idx'].values)

# # Define hyperparameters
# learning_rate = 0.01
# epochs = 10
# hidden_size = 12
# output_size = 4

# # Initialize the model, loss function, and optimizer
# net = Net(max_input_size, hidden_size, output_size)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=learning_rate)

# # Create dataloaders for the train and test datasets
# train_dataset = MyDataset(train_df)
# test_dataset = MyDataset(test_df)
# train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# # Train the model
# for epoch in range(epochs):
#     running_loss = 0.0
#     for i, data in enumerate(train_loader, 0):
#         inputs, labels = data
#         inputs = torch.tensor(inputs.tolist(), dtype=torch.float32)
#         labels = torch.tensor(labels.tolist(), dtype=torch.long)

#         optimizer.zero_grad()

#         outputs = net(inputs)

#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     print(f"Epoch {epoch+1} loss: {running_loss/len(train_loader)}")

# # Evaluate the model on the train and test sets
# with torch.no_grad():
#     train_pred = model(train_padded)
#     test_pred = model(test_padded)

#     # Get the predicted class for each set
#     train_pred_class = train_pred.argmax(dim=1)
#     test_pred_class = test_pred.argmax(dim=1)

#     # Calculate accuracy and print confusion matrix for train set
#     train_accuracy = accuracy_score(train_y, train_pred_class)
#     train_confusion_matrix = confusion_matrix(train_y, train_pred_class)
#     print("Train accuracy:", train_accuracy)
#     print("Train confusion matrix:\n", train_confusion_matrix)

#     # Calculate accuracy and print confusion matrix for test set
#     test_accuracy = accuracy_score(test_y, test_pred_class)
#     test_confusion_matrix = confusion_matrix(test_y, test_pred_class)
#     print("Test accuracy:", test_accuracy)
#     print("Test confusion matrix:\n", test_confusion_matrix)

