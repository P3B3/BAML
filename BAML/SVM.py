#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
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
    support = svm.SVC()
    X = df.values[:, 0:2]
    Y = df.values[:, 2]
    trainX, testX, trainY, testY = train_test_split( X, Y, test_size = 0.3)
    support.fit(trainX, trainY)

    print('Accuracy: \n', support.score(testX, testY))

    sns.set_context("notebook", font_scale=1.1)
    sns.set_style("ticks")
    sns.lmplot('X1','X2', scatter=True, fit_reg=False, data=df, hue='Y')
    plt.ylabel('X2')
    plt.xlabel('X1')


def getSVMpred(df,data):
    support = svm.SVC()
    X = df.values[:, 0:2]
    Y = df.values[:, 2]
    trainX, testX, trainY, testY = train_test_split( X, Y, test_size = 0.3)
    support.fit(trainX, trainY)
    pred = support.predict(data) 

    return pred


if __name__ == "__main__":
    path = 'iris_df.csv'
    set_columns = ['X4', 'X3', 'X1', 'X2', 'Y']
    drop_columns = ['X4', 'X3']

    displayAll(setData(path, set_columns, drop_columns))

    

