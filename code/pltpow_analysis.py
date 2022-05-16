import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score


###### the rounded regression model ############
class Rounded_Reg:
    """
    Given a linear regression model we convert it into a rounded_regression model rounding up to round_degree
    """
    def __init__(self, coeffs: list, intercept: float, round_degree: int):
        self.coeffs = np.array([round(coeff, round_degree) for coeff in coeffs])
        self.intercept = round(intercept, round_degree)

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            prediction = self.coeffs*np.array(X.iloc[i]) + self.intercept
            y_pred.append(prediction)
        return np.array(y_pred)
#################################################


################# rounded model creator #########
def get_rmodel(file, lower_epc, upper_epc, lower_dist, upper_dist):
    """
    Creates a rounded model for a pltpow file
    input:  lower_epc is the minimum epc a row must have to be included in the fit and test, similair for upper_epc. 
            lower_dist is the minimum dist a row must have to be incldued in the fit and test, similair for upper_dist

    output: returns (the coeff, the intercept, mean_absolute_error, mean_square_error, the R2 score)
    """
    df = pd.read_csv(file)
    df = df.loc[df['epc'] >= lower_epc]
    df = df.loc[df['epc'] <= upper_epc]
    df = df.loc[df['dist'] >= lower_dist]
    df = df.loc[df['dist'] <= upper_dist]
    df = df.loc[df['dist'] - round(df['dist']) == 0]
    df = df.loc[df['pow'] > .05]

    X = df[['dist']]
    y = df['pow']



    model = LinearRegression()
    model.fit(X,y)

    rmodel = Rounded_Reg(model.coef_, model.intercept_, 2)

    y_pred = rmodel.predict(X)
    m_a_error = mean_absolute_error(y, y_pred)
    m_2_error = mean_squared_error(y,y_pred)
    r2 = r2_score(y,y_pred)

    return (rmodel.coeffs, rmodel.intercept, m_a_error, m_2_error, r2)
