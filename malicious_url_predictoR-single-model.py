# -*- coding: utf-8 -*-
"""Malicious URL Predictor



In this project we will build, train, test, and evaluate a malicious URL predictor using the Scikit-learn machine learning library

**Outline:** 
* Initial Setup
* Tokenization
* Load Training Data
* Vectorize Training Data
* Load Testing Data
* Train and Evaluate Models

"""



  
# Installing required libraries




# Common imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import re
import pickle

# %matplotlib inline

# Import Scikit-learn helper functions
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Import Scikit-learn models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Import Scikit-learn metric functions
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns



print("\n### Libraries Imported ###\n")

"""# Load the dataset
With this set, we first need to load our CSV data.
"""

# Load the training data
print("- Loading CSV Data -")
url_df = pd.read_csv("Maliciousurl.csv")

test_url = url_df['URLs'][4]

print("\n### CSV Data Loaded ###\n")

# Let's see what our training data looks like
print(url_df)

# Perform Train/Test split
test_percentage = .2

train_df, test_df = train_test_split(url_df, test_size=test_percentage, random_state=42)

labels = train_df['Class']
test_labels = test_df['Class']

print("\n### Split Complete ###\n")

# Print counts of each class
print("- Counting Splits -")
print("Training Samples:", len(train_df))
print("Testing Samples:", len(test_df))

# Graph counts of each class, for both training and testing
count_train_classes = pd.value_counts(train_df['Class'])
count_train_classes.plot(kind='bar', fontsize=16)
plt.title("Class Count (Training)", fontsize=20)
plt.xticks(rotation='horizontal')
plt.xlabel("Class", fontsize=20)
plt.ylabel("Class Count", fontsize=20)

plt.show()

count_test_classes = pd.value_counts(test_df['Class'])
count_test_classes.plot(kind='bar', fontsize=16, colormap='ocean')
plt.title("Class Count (Testing)", fontsize=20)
plt.xticks(rotation='horizontal')
plt.xlabel("Class", fontsize=20)
plt.ylabel("Class Count", fontsize=20)

plt.show()

"""# Tokenization
Create our tokenizer by splitting URLs into their domains, subdomains, directories, files, and extensions.
"""

# Define tokenizer
#   The purpose of a tokenizer is to separate the features from the raw data


def tokenizer(url):
  """Separates feature words from the raw data
  Keyword arguments:
    url ---- The full URL
    
  :Returns -- The tokenized words; returned as a list
  """
  
  # Split by slash (/) and dash (-)
  tokens = re.split('[/-]', url)
  
  for i in tokens:
    # Include the splits extensions and subdomains
    if i.find(".") >= 0:
      dot_split = i.split('.')
      
      # Remove .com and www. since they're too common
      if "com" in dot_split:
        dot_split.remove("com")
      if "www" in dot_split:
        dot_split.remove("www")
      
      tokens += dot_split
      
  return tokens
    
print("\n### Tokenizer defined ###\n")

## Task 1 - Tokenize a URL


# Let's see how our tokenizer changes our URLs

print("\n- Full URL -\n")

print(test_url)



# Tokenize test URL
print("\n- Tokenized Output -\n")

tokenized_url = tokenizer(test_url)
print(tokenized_url)

"""# Vectorize the Data
Now that the training data has been loaded, we'll train the vectorizers to turn our features into numbers.

"""



print("- Training Count Vectorizer -")


cVec = CountVectorizer(tokenizer=tokenizer)
count_X = cVec.fit_transform(train_df['URLs'])



print("- Training TF-IDF Vectorizer -")

tVec = TfidfVectorizer(tokenizer=tokenizer)
tfidf_X = tVec.fit_transform(train_df['URLs'])



# (Keep the following lines)
print("\n### Vectorizing Complete ###\n")

"""## Task 2a  - Count the test URL tokens

"""

# Manually perform term count on test_url

for token in list(dict.fromkeys(tokenized_url)):
  print("{} - {}".format(tokenized_url.count(token), token))

"""## Task 2b  - View the test URL vectorizers

"""

print("\n- Count Vectorizer (Test URL) -\n")

exvec = CountVectorizer(tokenizer=tokenizer)
exx = exvec.fit_transform([test_url])
print(exx)


# (Keep the following lines)
print()
print("=" * 50)
print()




print("\n- TFIDF Vectorizer (Test URL) -\n")


exvec = TfidfVectorizer(tokenizer=tokenizer)
exx = exvec.fit_transform([test_url])
print(exx)

"""# Test and Evaluate the Models
"""

# Vectorize the testing inputs
#   Use 'transform' instead of 'fit_transform' because we've already trained our vectorizers

print("- Count Vectorizer -")


test_count_X = cVec.transform(test_df['URLs'])



print("- TFIDF Vectorizer -")


test_tfidf_X = tVec.transform(test_df['URLs'])


print("\n### Vectorizing Complete ###\n")

#report generator

def generate_report(cmatrix, score, creport):
  """Generates and displays graphical reports
  """
  
  # Transform cmatrix because Sklearn has pred as columns and actual as rows.
  cmatrix = cmatrix.T
  
  # Generate confusion matrix heatmap
  plt.figure(figsize=(5,5))
  sns.heatmap(cmatrix, 
              annot=True, 
              fmt="d", 
              linewidths=.5, 
              square = True, 
              cmap = 'Blues', 
              annot_kws={"size": 16}, 
              xticklabels=['bad', 'good'],
              yticklabels=['bad', 'good'])

  plt.xticks(rotation='horizontal', fontsize=16)
  plt.yticks(rotation='horizontal', fontsize=16)
  plt.xlabel('Actual Label', size=20);
  plt.ylabel('Predicted Label', size=20);

  title = 'Accuracy Score: {0:.4f}'.format(score)
  plt.title(title, size = 20);

  # Display classification report and confusion matrix
  print(creport)
  plt.show()
  

print("\n### Report Generator Defined ###\n")

# Logistic Regression with Count Vectorizer

# Train the model


lgs_count = LogisticRegression(solver='lbfgs')
lgs_count.fit(count_X, labels)



# Test the mode (score, predictions, confusion matrix, classification report)


score_lgs_count = lgs_count.score(test_count_X, test_labels)
predictions_lgs_count = lgs_count.predict(test_count_X)
cmatrix_lgs_count = confusion_matrix(test_labels, predictions_lgs_count)
creport_lgs_count = classification_report(test_labels, predictions_lgs_count)





# (Keep the following lines)
print("\n### Model Built ###\n")
generate_report(cmatrix_lgs_count, score_lgs_count, creport_lgs_count)


pickle.dump(lgs_count,open('model.pkl','wb'))
