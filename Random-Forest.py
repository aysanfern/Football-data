#importing relevant libraries
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor



os.chdir("C:\\Users\\aysan\\Desktop\\fifa19") #changing working directory to the directory with the fifa 19 data

df = pd.read_csv('data.csv') #importing the data

df.head() #looking at the dat

df.fillna(0,inplace=True) #fills all cells that do not have any entries with zero
df.info() #looks at each column and shows the data type inside that column

#pd.options.display.max_columns=89 #let's you see all the columns in the dataset when printed below
#df.head()

#making the value column into an array to perform some functions before importing it back into the column
array=df.Value.tolist() #extracts the 'value' column as a list
lst=[]
for i in range(len(array)):
    if 'K' in array[i]:
        lst.append(i) #this will add all the index's that are in K format into the 'lst' list
        for i in range(len(lst)):
            array[lst[i]]=array[lst[i]].strip('€K') # takes away the currency and the suffix so we can perform operations on it
    else:
        array[i]=array[i].strip('€M')
        array[i]=float(array[i]) #formats the values as a float instead of a string
for i  in range(len(lst)):
    array[lst[i]]=float(array[lst[i]])/1000 #divides the values that are denoted with a 'K' suffix by 1000 so value unit is M

df.Value=array #importing the array back into the column


X=df[['Age','Value']].values #these are the features we are going to use to predict Potential
y=df['Potential'].values


regressor = RandomForestRegressor(n_estimators= 100)
regressor.fit(X,y)



xt=[37,1]
xtx=np.asarray(xt).reshape(1,-1)
yp=regressor.predict(xtx)

print(yp)


