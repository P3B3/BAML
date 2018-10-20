#%%
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import warnings
import graphviz 
from graphviz import Source

warnings.filterwarnings("ignore")


def setData(dataSetPath,setColumns,setDrop):
    df = pd.read_csv(dataSetPath)
    if setColumns != None:
        df.columns = setColumns
    if setDrop != None:
        df = df.drop(setDrop, 1)
        
    return df


def drawTree(df, crit):
    decision = tree.DecisionTreeClassifier(criterion='gini')
    X = df.values[:, 0:4]
    Y = df.values[:, 4]
    trainX, testX, trainY, testY = train_test_split( X, Y, test_size = 0.3)
    decision.fit(trainX, trainY)
    
    print('Accuracy: \n', decision.score(testX, testY))

    dot_data = tree.export_graphviz(decision, out_file=None)
    graph = graphviz.Source(dot_data).view("iris")


def score(df, crit):
    decision = tree.DecisionTreeClassifier(criterion='gini')
    X = df.values[:, 0:4]
    Y = df.values[:, 4]
    trainX, testX, trainY, testY = train_test_split( X, Y, test_size = 0.3)
    decision.fit(trainX, trainY)
    
    return decision.score(testX, testY)
    


if __name__ == "__main__":
    path = 'iris_df.csv'
    set_columns = ['X1', 'X2', 'X3', 'X4', 'Y']
    drop_columns = None
    crit = 'gini'

    drawTree(setData(path, set_columns, drop_columns), crit)



    


