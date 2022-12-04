# Mini-Project--Application-of-NN
## Project Title:
### Rainfall Prediction.
## Project Description
Rainfall Prediction is the application of science and technology to predict the amount of rainfall over a region. It is important to exactly determine the rainfall for effective use of water resources, crop productivity and pre-planning of water structures.
## Algorithm:
1.Import necessary libraries.

2.Apply the rainfall dataset to algoritm.

3.Read the dataset.

4.Plot the graph and correlation matrix.

5.Study the final output.

## Program:

Developed By Team Members:
1.Manoj Guna Sundar Tella.
2.P.Siva Naga Nithin.
3.D.Amarnath Reddy.



from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] 
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()


def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') 
    df = df[[col for col in df if df[col].nunique() > 1]]
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()


def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number])
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] 
    columnNames = list(df)
    if len(columnNames) > 10: 
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()


nRowsRead = 1000
df2 = pd.read_csv('rainfall.csv', delimiter=',', nrows = nRowsRead)
df2.dataframeName = 'rainfall in india 1901-2015.csv'
nRow, nCol = df2.shape
print(f'There are {nRow} rows and {nCol} columns')


df2.head(5)


plotPerColumnDistribution(df2, 10, 5)


plotCorrelationMatrix(df2, 8)


plotScatterMatrix(df2, 20, 10)

## Output:
![nnp1](https://user-images.githubusercontent.com/94883876/205504768-b658c180-f751-43c9-94b8-dc4f5efd2a04.jpg)
![nnp2](https://user-images.githubusercontent.com/94883876/205504779-c830234e-0a49-41e6-bf75-5a3162f9e37c.jpg)
![nnp3](https://user-images.githubusercontent.com/94883876/205504793-fae886ba-7d62-49f9-a0d9-772cbf8082f5.jpg)
![nnp4](https://user-images.githubusercontent.com/94883876/205504803-81ba4da5-4b86-40a4-88a1-df512eaca496.jpg)
![nnp5](https://user-images.githubusercontent.com/94883876/205504816-1e3fd577-edcc-4214-bc23-5091c2c4ce40.jpg)



## Advantage :
Rainfall prediction is important as heavy rainfall can lead to many disasters. The prediction helps people to take preventive measures and moreover the prediction should be accurate. There are two types of prediction short term rainfall prediction and long term rainfall.
## Result:
Thus Implementation of Rainfall Prediction was executed successfully.
