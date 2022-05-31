import pandas as pd
import numpy as np

df = pd.read_csv('EpcProject3.csv')

class Evaluator:
    """
    Class for evaluating the preformance of manually made models. Input the variance you want to test for
    """
    def __init__(self, variance = None) -> None:
        if type(variance) != None:
            self.df = df.loc[df['Description'] == variance]
        else:
            self.df = df
    
    def accuracy_score(self, predictions, y):
        current_correct = 0 
        for i,prediction in np.ndenumerate(predictions):
            if prediction == y.iloc[i]:
                current_correct += 1
        return current_correct/len(y)
    
    def dummy_score(self):
        high_rows = df[df['High'] == True]
        percentage = high_rows.shape[0]/df.shape[0]
        

def high_prediction(X):
    """
    manual made model for predicting when a position is high or not high
    """
    predictions = np.array([])
    for x in np.nditer(X):
        if -2*x + 1 >= 0:
            predictions = np.append(predictions, [True])
        else: 
            predictions = np.append(predictions, [False])
    return predictions

def high_pred_easy(X):
    """
    Attempted simplification of high_model
    """
    predictions = np.array([])
    for x in np.nditer(X):
        if x <= 1:
            predictions = np.append(predictions, [True])
        else:
            predictions = np.append(predictions, [False])
    return predictions