import numpy as np
import csv
import sys
import json
import math
from train import preprocessing
from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path', using the model learned during training.
Writes the predicted values to the file named "predicted_test_Y_nb.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""


def import_data_and_model(test_X_file_path, model_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter='\n', dtype=str)
    #model = json.load(open(model_file_path).read())
    with open('model_file.json') as f:
        model=json.load(f)
    return test_X, model



def compute_likelihood(test_X, c,class_wise_frequency_dict,class_wise_denominators):
    #TODO Complete the function implementation. Read the Question text for details
    test_X=test_X.split(' ')
    #print(test_X,c)
    calculate_likelihood=0
    #print(class_wise_frequency_dict[c]['had'])
    for x in test_X:
       calculate_likelihood+=np.log((class_wise_frequency_dict[c].get(x,0)+1)/class_wise_denominators[c])
       
    return calculate_likelihood

def predict_data(test_X,classes,class_wise_frequency_dict, class_wise_denominators,prior_probabilities):
    #TODO Complete the function implementation. Read the Question text for details
    max=-math.inf
    predict=-math.inf
    for x in classes:
       if max<compute_likelihood(test_X,x,class_wise_frequency_dict, class_wise_denominators)+np.log(prior_probabilities[x]):
           max=compute_likelihood(test_X,x,class_wise_frequency_dict, class_wise_denominators)+np.log(prior_probabilities[x])
           predict=x
    
    return predict

def predict_target_values(test_X, model):
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.
    #print(test_X,model)
    new_test_X=[]
    classes = np.genfromtxt('train_Y_nb.csv', delimiter='\n', dtype=str)
    for x in test_X:
        new_test_X.append(preprocessing(x))
    test_X=np.array(new_test_X)
    dict_word_frequency,prior_probability,class_wise_denominator=model[0],model[1],model[2]
    classes = list(set(classes))
    classes.sort()
    #print(classes)
    result=[]
    for x in test_X:
       result.append(predict_data(x,classes,dict_word_frequency,class_wise_denominator,prior_probability))
    print(result)
    return np.array(result)

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    # Load Model Parameters
    """
    Parameters of Naive Bayes include Laplace smoothing parameter, Prior probabilites of each class and values related to likelihood computation.
    """
    test_X, model = import_data_and_model(test_X_file_path, "./model_file.json")
    pred_Y = predict_target_values(test_X, model)
    write_to_csv_file(pred_Y, "predicted_test_Y_nb.csv")    


if __name__ == "__main__":
    test_X_file_path =sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_nb.csv") 
