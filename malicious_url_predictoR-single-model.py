# -*- coding: utf-8 -*-
"""Malicious URL Predictor



**Goal:** This workbook will walk you through the steps to build, train, test, and evaluate a malicious URL predictor using the Scikit-learn machine learning library

**Outline:** 
* Initial Setup
* Tokenization
* Load Training Data
* Vectorize Training Data
* Load Testing Data
* Train and Evaluate Models


# Initial Setup
We'll start by importing the data and loading the needed libraries.
"""



  
# Install dependencies




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

"""## Task 1 - Tokenize a URL
1. Print the full URL, **test_url**
2. Print the results of **tokenizer(test_url)**
"""

# Let's see how our tokenizer changes our URLs

print("\n- Full URL -\n")
# (Write code here)
print(test_url)



# Tokenize test URL
print("\n- Tokenized Output -\n")
# (Write code here)
tokenized_url = tokenizer(test_url)
print(tokenized_url)

"""# Vectorize the Data
Now that the training data has been loaded, we'll train the vectorizers to turn our features into numbers.

## Task 2 - Train the vectorizers
1. Create the count vectorizer **cVec** using the **CountVectorizer** function
2. Configure *cVec* to use the *tokenizer* function from earlier
3. Perform **fit_transform** on *cVec* to train the vectorizer with the *training URLs*\
a. Save the result as **count_X**


4. Create the TF-IDF vectorizer **tVec** using the **TfidfVectorizer** function
5. Configure *tVec* to use the *tokenizer* function from earlier
6. Perform **fit_transform** on *tVec* to train the vectorizer with the *training URLs*\
a. Save the result as **tfidf_X**
"""

# Vectorizer the training inputs -- Takes about 30 seconds to complete
#   There are two types of vectors:
#     1. Count vectorizer
#     2. Term Frequency-Inverse Document Frequency (TF-IDF)

print("- Training Count Vectorizer -")
# (Write code here)

cVec = CountVectorizer(tokenizer=tokenizer)
count_X = cVec.fit_transform(train_df['URLs'])



print("- Training TF-IDF Vectorizer -")
# (Write code here)

tVec = TfidfVectorizer(tokenizer=tokenizer)
tfidf_X = tVec.fit_transform(train_df['URLs'])



# (Keep the following lines)
print("\n### Vectorizing Complete ###\n")

"""## Task 2a (optional) - Count the test URL tokens
1. Print the count of each *token* from **test_url**
"""

# Manually perform term count on test_url
# (Write code here)
for token in list(dict.fromkeys(tokenized_url)):
  print("{} - {}".format(tokenized_url.count(token), token))

"""## Task 2b (optional) - View the test URL vectorizers
1. Create a new **CountVectorizer** and **TfidfVectorizer** for demonstration
2. Train the new vectorizers on **test_url** using **fit_transform**
3. Print the results of each *transform*
"""

print("\n- Count Vectorizer (Test URL) -\n")
# (Write code here)
exvec = CountVectorizer(tokenizer=tokenizer)
exx = exvec.fit_transform([test_url])
print(exx)


# (Keep the following lines)
print()
print("=" * 50)
print()





print("\n- TFIDF Vectorizer (Test URL) -\n")
# (Write code here)

exvec = TfidfVectorizer(tokenizer=tokenizer)
exx = exvec.fit_transform([test_url])
print(exx)

"""# Test and Evaluate the Models
OK, we have our training data loaded and our testing data loaded. Now it's time to train and evaluate our models.

But first, we're going to define a helper function to display our evaluation reports.

## Task 3 - Vectorize the testing data
1. Use **cVec** to *transform* **test_df['URLs']**\
a. Save the result as **test_count_X**

2. Use **tVec** to *transform* **test_df['URLs']**\
a. Save the result as **test_tfidf_X**
"""

# Vectorize the testing inputs
#   Use 'transform' instead of 'fit_transform' because we've already trained our vectorizers

print("- Count Vectorizer -")
# (Write code here)

test_count_X = cVec.transform(test_df['URLs'])



print("- TFIDF Vectorizer -")
# (Write code here)

test_tfidf_X = tVec.transform(test_df['URLs'])


print("\n### Vectorizing Complete ###\n")

# Define report generator

def generate_report(cmatrix, score, creport):
  """Generates and displays graphical reports
  Keyword arguments:
    cmatrix - Confusion matrix generated by the model
    score --- Score generated by the model
    creport - Classification Report generated by the model
    
  :Returns -- N/A
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



"""## Task 4d - Train and evaluate the LGS-Count model
1. Create **lgs_count** as a **LogisticRegression()** constructor, using the **lbfgs** *solver*
2. Use **fit** to train *lgs_count* on the training data (*count_X*) and training labels (*labels*)
3. Evaluate the model with the testing data (*test_count_X*) and testing labels (*test_labels*):\
a. Use the **score** function in *lgs_count* to calculate model accuracy; save the results as **score_lgs_count**\
b. Use the **predict** function in *lgs_count* to generate model predictions; save the results as **predictions_lgs_count**\
c. Generate the confusion matrix with **confusion_matrix**, using the predictons and labels; save the results as **cmatrix_lgs_count**\
d. Generate the classification report with **classification_report**, using the predictions and labels; save the results as **creport_lgs_count**
"""

# Logistic Regression with Count Vectorizer

# Train the model
# (Write code here)

lgs_count = LogisticRegression(solver='lbfgs')
lgs_count.fit(count_X, labels)



# Test the mode (score, predictions, confusion matrix, classification report)
# (Write code here)

score_lgs_count = lgs_count.score(test_count_X, test_labels)
predictions_lgs_count = lgs_count.predict(test_count_X)
cmatrix_lgs_count = confusion_matrix(test_labels, predictions_lgs_count)
creport_lgs_count = classification_report(test_labels, predictions_lgs_count)





# (Keep the following lines)
print("\n### Model Built ###\n")
generate_report(cmatrix_lgs_count, score_lgs_count, creport_lgs_count)


pickle.dump(lgs_count,open('model.pkl','wb'))
