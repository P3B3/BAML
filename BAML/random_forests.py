#%%
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
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


def getRndForest(df):
    forest = RandomForestClassifier()
    X = df.values[:, 0:4]
    Y = df.values[:, 4]
    trainX, testX, trainY, testY = train_test_split( X, Y, test_size = 0.3)
    forest.fit(trainX, trainY)

    print('Accuracy: ', forest.score(testX, testY))

    pred = forest.predict(testX)
    return pred


if __name__ == "__main__":
    path = 'iris_df.csv'
    set_columns = ['X1', 'X2', 'X3', 'X4', 'Y']
    drop_columns = None

    getRndForest(setData(path, set_columns, drop_columns))