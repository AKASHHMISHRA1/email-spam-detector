# -*- coding: utf-8 -*-
import numpy as np
import csv
import sys
import json
from collections import Counter

from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path', using the model learned during training.
Writes the predicted values to the file named "predicted_test_Y_nb.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_dataset(train_X_file_path, train_Y_file_path):
    train_X = np.genfromtxt(train_X_file_path, delimiter='\n', dtype=str)
    train_Y = np.genfromtxt(train_Y_file_path, delimiter='\n', dtype=str)
    return train_X ,train_Y

def preprocessing(s):
    #TODO Complete the function implementation. Read the Question text for details
    s=s.lower()
    #print(s)
    new_s=''
    prev_space=1
    for x in s:
         if x.islower():
                new_s+=x
                prev_space=0
         elif x.isspace():
                if prev_space==1:
                    continue
                else:
                    new_s+=' '
                prev_space=1        
    return new_s

def listToString(X):
    str=' '
    return str.join(X)

def sorted_word_list(string_class):
    words=string_class.split()
    words.sort()
    return words
    
def class_wise_words_frequency_dict(X, Y):
    #TODO Complete the function implementation. Read the Question text for details
    #print(X,Y)
    Y_=np.array(Y)
    X_=np.array(X)
    class_labels=list(set(Y))
    this_dict={}
    for x in class_labels:
        #print(x)
        string_class=listToString(X_[np.where(Y_==x,True,False)])
        words=sorted_word_list(string_class)
        #print(words)
        dict_words=Counter(words)
        #print(dict_words)
        this_dict[x]=dict_words
    return this_dict

def compute_prior_probabilities(Y):
    classes = list(set(Y))
    n_docs = len(Y)
    prior_probabilities = dict()
    for c in classes:
        prior_probabilities[c] = Y.count(c) / n_docs
    return prior_probabilities

def get_class_wise_denominators_likelihood(X, Y):
    #TODO Complete the function implementation. Read the Question text for details
    #print(X,Y)
    Y_=np.array(Y)
    X_=np.array(X)
    class_labels=list(set(Y))
    this_dict={}
    vocabulary=sorted_word_list(listToString(X_))
    vocabulary_len=len(set(vocabulary))
    for x in class_labels:
        #print(x)
        string_class=listToString(X_[np.where(Y_==x,True,False)])
        #print(string_class)
        words=sorted_word_list(string_class)
        this_dict[x]=len(words)+vocabulary_len
        
    return this_dict

def train_model(train_X, train_Y):
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.
    #print(train_X,train_Y)
    #print(train_X.shape)
    new_train_X=[]
    for x in train_X:
        new_train_X.append(preprocessing(x))
    train_X=np.array(new_train_X)
    print(train_X.shape)
    dict_word_frequency=class_wise_words_frequency_dict(train_X,train_Y)
    #print(dict_word_frequency)
    prior_probability=compute_prior_probabilities(list(train_Y))
    print(prior_probability)
    class_wise_denominator=get_class_wise_denominators_likelihood(train_X,train_Y)
    print(class_wise_denominator)
    return dict_word_frequency,prior_probability,class_wise_denominator
    

def write_to_json_file(dict_word_frequency,prior_probability,class_wise_denominator):
    model=[dict_word_frequency,prior_probability,class_wise_denominator]
    print(model)
    with open('model_file.json', 'w') as json_file:
      json.dump(model, json_file, indent=4)


def train(train_X_file_path,train_Y_file_path):
    # Load Model Parameters
    """
    Parameters of Naive Bayes include Laplace smoothing parameter, Prior probabilites of each class and values related to likelihood computation.
    """
    train_X, train_Y = import_dataset(train_X_file_path,train_Y_file_path)
    dict_word_frequency,prior_probability,class_wise_denominator= train_model(train_X,train_Y)
    write_to_json_file(dict_word_frequency,prior_probability,class_wise_denominator)    


if __name__ == "__main__":
    train_X_file_path ='train_X_nb.csv'
    train_Y_file_path='train_Y_nb.csv'
    train(train_X_file_path,train_Y_file_path)
    # Uncomment to test on the training data
    # validate(test_X_file_path, actual_test_Y_file_path="train_Y_nb.csv") 
