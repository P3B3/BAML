#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")


def setData(dataSetPath,setColumns,setDrop):
    df = pd.read_csv(dataSetPath)
    if setColumns != None:
        df.columns = setColumns
    if setDrop != None:
        df = df.drop(setDrop, 1)
        
    return df


def displayAll(df):
    sns.set_context("notebook", font_scale=1.1)
    sns.set_style("ticks")
    sns.regplot('X','Y', data=df, logistic=True)
    plt.ylabel('Probability')
    plt.xlabel('Explanatory')

    logistic = LogisticRegression()
    X = (np.asarray(df.X)).reshape(-1, 1)
    Y = (np.asarray(df.Y)).ravel()
    logistic.fit(X, Y)
    logistic.score(X, Y)

    print('Coefficient: \n', logistic.coef_)
    print('Intercept: \n', logistic.intercept_)
    print('R^2 Value: \n', logistic.score(X, Y))


def getLogRegValues(df):
    logistic = LogisticRegression()
    X = (np.asarray(df.X)).reshape(-1, 1)
    Y = (np.asarray(df.Y)).ravel()
    logistic.fit(X, Y)
    logistic.score(X, Y)
    values = {'coefficient': logistic.coef_, 'intercept': logistic.intercept_, 'r^2': logistic.score(X, Y)}

    return values

if __name__ == "__main__":
    path = 'logistic_regression_df.csv'
    set_columns = ['X', 'Y']
    drop_columns = None

    displayAll(setData(path, set_columns, drop_columns))

