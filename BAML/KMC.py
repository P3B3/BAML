#%%
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from IPython.display import display
import warnings

warnings.filterwarnings("ignore")


def setData(dataSetPath,setColumns,setDrop):
    df = pd.read_csv(dataSetPath)
    if setColumns != None:
        df.columns = setColumns
    if setDrop != None:
        df = df.drop(setDrop, 1)
        
    return df


def displayAll(df,cluster_size):
    kmeans = KMeans(n_clusters=cluster_size)
    X = df.values[:, 0:2]
    kmeans.fit(X)
    df['Pred'] = kmeans.predict(X)
    display(df.head())

    sns.set_context("notebook", font_scale=1.1)
    sns.set_style("ticks")
    sns.lmplot('X1','X2', scatter=True, fit_reg=False, data=df, hue = 'Pred')


def getKmeansPred(df,cluster_size):
    kmeans = KMeans(n_clusters=cluster_size)
    X = df.values[:, 0:2]
    kmeans.fit(X)
    df['Pred'] = kmeans.predict(X)
    
    return df


if __name__ == "__main__":
    path = 'iris_df.csv'
    set_columns = ['X1', 'X2', 'X3', 'X4', 'Y']
    drop_columns = ['X4', 'X3']
    cluster_size = 3

    displayAll(setData(path, set_columns, drop_columns), cluster_size)