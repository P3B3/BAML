#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import linear_model
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
    sns.lmplot('X','Y', data=df)
    plt.ylabel('Response')
    plt.xlabel('Explanatory')

    linear = linear_model.LinearRegression()
    trainX = np.asarray(df.X[800:len(df.X)]).reshape(-1, 1)
    trainY = np.asarray(df.Y[800:len(df.Y)]).reshape(-1, 1)
    linear.fit(trainX, trainY)
    linear.score(trainX, trainY)

    print('Coefficient: ', linear.coef_)
    print('Intercept: ', linear.intercept_)
    print('R^2 Value: ', linear.score(trainX, trainY))


def getLineRegPred(df,data):
    linear = linear_model.LinearRegression()
    trainX = np.asarray(df.X[800:len(df.X)]).reshape(-1, 1)
    trainY = np.asarray(df.Y[800:len(df.Y)]).reshape(-1, 1)
    linear.fit(trainX, trainY)
    linear.score(trainX, trainY)
    predicted = linear.predict(data)

    return predicted


if __name__ == "__main__":
    path = 'linear_regression_df.csv'
    set_columns = None
    drop_columns = None
    
    displayAll(setData(path, set_columns,drop_columns))