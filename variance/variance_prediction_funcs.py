import pandas as pd
import numpy as np

df = pd.read_csv('EpcProject3.csv')

class Evaluator:
    """
    Class for evaluating the preformance of manually made models. Input the variance you want to test for
    """
    def __init__(self, variance = None) -> None:
        self.variance = variance
        if type(variance) != None:
            self.df = df.loc[df['Description'] == variance]
        else:
            self.df = df
    
    def accuracy_score(self, model, params):
        df_roll = df.loc[df['Description'] == 'Rollish']
        df_low = df.loc[df['Description'] == 'Low']
        df_med = df.loc[df['Description'] == 'Medium']
        df_high = df.loc[df['Description'] == 'High']
        n = min(df_roll.shape[0], df_low.shape[0], df_med.shape[0], df_high.shape[0])
        df_roll = df.loc[df['Description'] == 'Rollish'].sample(n = n)
        df_low = df.loc[df['Description'] == 'Low'].sample(n = n)
        df_med = df.loc[df['Description'] == 'Medium'].sample(n = n)
        df_high = df.loc[df['Description'] == 'High'].sample(n = n)
        df_eval = pd.concat([df_roll, df_low, df_med, df_high])

        X = df_eval[params]
        y = df_eval[self.variance]
        predictions = model(X)
        current_correct = 0 
        for i in range(predictions.shape[0]):
            prediction = predictions[i]
            if prediction == y.iloc[i]:
                current_correct += 1
        return current_correct/len(y)
    
    def dummy_score(self):
        high_rows = df[df['High'] == True]
        percentage = high_rows.shape[0]/df.shape[0]
        return percentage
        
########################## high variance  ##############################
def high_prediction(X):
    """
    manual made model for predicting when a position is high or not high
    """
    predictions = np.array([])
    for i, x in np.ndenumerate(X):
        if -2.7*x + 1 >= 0:
            predictions = np.append(predictions, [True])
        else: 
            predictions = np.append(predictions, [False])
    return predictions

def high_pred_hard(X):
    """
    better_high_model
    """
    predictions = np.array([])
    for i in range(X.shape[0]):
        data_point = X.iloc[i]
        ad = data_point[0]
        three = data_point[1]
        if -3*ad + -1*three + 4 >= 0:
            predictions = np.append(predictions, [True])
        else:
            predictions = np.append(predictions, [False])
    return predictions
#####################################################################


###################### rollish variance #############################
def roll_pred(X):
    """
    rollish model
    """
    predictions = np.array([])
    for i in range(X.shape[0]):
        data_point = X.iloc[i]
        ad = data_point[0]
        pc = data_point[1]
        if .6*ad + -.1*pc -1.5 >= 0:
            predictions = np.append(predictions, [True])
        else:
            predictions = np.append(predictions, [False])
    return predictions