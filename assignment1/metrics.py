import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = recall = accuracy = f1 = 0

    FN = len(np.where((ground_truth == True) & (prediction == False))[0]) # false negatives;
    TN = len(np.where((ground_truth == False) & (prediction == False))[0]) # true negatives; 
    TP = len(np.where((ground_truth == True) & (prediction == True))[0]) # true positives;   
    FP = len(np.where((ground_truth == False) & (prediction == True))[0])  # false positives;       
    
    precision = TP / (TP + FP)                  # как много из извлечённых элементов релевантны?
    recall    = TP / (TP + FN)                  # как много из релевантных элементов получено?
    accuracy  = (TP + TN) / (TP + TN + FP + FN) # общая точность;
    f1        = 2 * ((precision * recall) / (precision + recall)) 
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    return len(np.where(prediction == ground_truth)[0]) / len(prediction)