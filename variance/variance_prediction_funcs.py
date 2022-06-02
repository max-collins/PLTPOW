import pandas as pd
import numpy as np
rng = np.random.default_rng(12345)
df_0 = pd.read_csv('EpcProject3.csv')

class Evaluator:
    """
    Class for evaluating the preformance of manually made models. Input the variance you want to test for
    """
    def __init__(self, variance = None, df = df_0)-> None:
        self.variance = variance
        # if type(variance) != None:
        #     self.df = df.loc[df['Description'] == variance]
        # else:
        self.df = df
    
    def head(self):
        return self.df.head()
    
    def accuracy_score(self, model, params, balanced = False):
        if balanced == True:
            df_roll = self.df.loc[self.df['Description'] == 'Rollish']
            df_low = self.df.loc[self.df['Description'] == 'Low']
            df_med = self.df.loc[self.df['Description'] == 'Medium']
            df_high = self.df.loc[self.df['Description'] == 'High']
            n = min(df_roll.shape[0], df_low.shape[0], df_med.shape[0], df_high.shape[0])
            df_roll = self.df.loc[self.df['Description'] == 'Rollish'].sample(n = n)
            df_low = self.df.loc[self.df['Description'] == 'Low'].sample(n = n)
            df_med = self.df.loc[self.df['Description'] == 'Medium'].sample(n = n)
            df_high = self.df.loc[self.df['Description'] == 'High'].sample(n = n)
            df_eval = pd.concat([df_roll, df_low, df_med, df_high])
        else:
            df_eval = self.df

        X = df_eval[params]    
        y = df_eval[self.variance]

        if model == dummy_model:
            predictions = model(X, self.variance)
        else:
            predictions = model(X)
        current_correct = 0 
        for i in range(predictions.shape[0]):
            prediction = predictions[i]
            if prediction == y.iloc[i]:
                current_correct += 1
        return current_correct/self.df.shape[0]

    def misses(self, model, params):
        df_eval = self.df
        misses = []

        X = df_eval[params]
        y = df_eval[self.variance]
        predictions = model(X)

        for i in range(predictions.shape[0]):
            prediction = predictions[i]
            if prediction != y.iloc[i]:
                misses.append((self.df.iloc[i]['Position'], self.df.iloc[i]['Description']))
        return misses

######################## dummy model just to make sure #################
def dummy_model(X, variance):
    var_rows = df[df[variance] == True]
    percentage = var_rows.shape[0]/df.shape[0]

    predictions = np.array([])
    for i in range(X.shape[0]):
        roll = rng.random()
        if roll <= percentage:
            predictions = np.append(predictions, [True])
        else: 
            predictions = np.append(predictions, [False])
    return predictions
########################################################################


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
    better high model
    """
    predictions = np.array([])
    for i in range(X.shape[0]):
        data_point = X.iloc[i]
        ad = data_point[0]
        three = data_point[1]
        if -2.5*ad + -1*three + 3 >= 0:
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

