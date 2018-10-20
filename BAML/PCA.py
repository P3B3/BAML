#%%
import pandas as pd
from sklearn import decomposition
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


def getPCA(df):  
    pca = decomposition.PCA()
    fa = decomposition.FactorAnalysis()
    X = df.values[:, 0:4]
    Y = df.values[:, 4]
    train, test = train_test_split(X,test_size = 0.3)
    train_reduced = pca.fit_transform(train)
    test_reduced = pca.transform(test)
    pcaValue = pca.n_components_

    return pcaValue


if __name__ == "__main__":
    path = 'iris_df.csv'
    set_columns = ['X1', 'X2', 'X3', 'X4', 'Y']
    drop_columns = None
    
    print (getPCA(setData(path, set_columns, drop_columns)))