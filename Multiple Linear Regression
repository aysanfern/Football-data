
#importing relevant libraries
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split




os.chdir() #changing working directory to the directory with the fifa 19 data



df = pd.read_csv('data.csv') #importing the data




df.fillna(method='bfill',inplace=True) #fills all cells that do not have any entries with zero
df.info() #looks at each column and shows the data type inside that column




'''pd.options.display.max_columns=89 #let's you see all the columns in the dataset when printed below
df.head()'''

df['Release Clause']=df['Release Clause'].astype(str).str.replace('\D+', '')
array2=df['Release Clause'].tolist()


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





type(array2[1])





type(array2[1])





array2=df['Release Clause'].tolist()

lst1=[]
for i in range(len(array2)):
    if 'K' in array2[i]:
        lst1.append(i) #this will add all the index's that are in K format into the 'lst' list
        for i in range(len(lst1)):
            array2[lst1[i]]=array[lst1[i]].strip('€K') # takes away the currency and the suffix so we can perform operations on it
    else:
        array2[i]=array2[i].strip('€M')
        array2[i]=float(array2[i])#formats the values as a float instead of a string

for i  in range(len(lst1)):
    array2[lst1[i]]=float(array2[lst1[i]])/1000 #divides the values that are denoted with a 'K' suffix by 1000 so value unit is M




df['Release Clause']=array2





df['Wage']=df['Wage'].apply(lambda x: x.strip('€K'))
df['Wage']=df['Wage'].apply(lambda x:float(x))





simple =LinearRegression()
X=df[['Age','Value','Release Clause','Wage']].values #these are the features we are going to use to predict Potential
y=df.iloc[:,8:9].values





df.head()





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=40) #splitting the data set
print(X_train)











simple.fit(X_train,y_train) #fitting the trained values to create our regression model 





#plotting a graph however with multilinear regression this won't be helpful for explicity feature relationships
'''
y_pred=simple.predict(X_test)
plt.plot(X_test,y_pred)
plt.ylabel('Potential')
plt.xlabel('Age (Decreasing)')
plt.title('Age vs Potential')
'''





''' y_pred=simple.predict(X_test)
#plt.plot(X_test,y_pred)
#plt.ylabel('Potential')
#plt.xlabel('Age (Decreasing)')
#plt.title('Age vs Potential')'''





x_theory=[28 ,120,50, 1000] #inputed value to predict potential (age,value(M))
xt=np.asarray(x_theory).reshape(1,-1) #formatting the values so they can be plugged into the regression

y_pred=simple.predict(xt)





print(y_pred) #seeing how badly a multivariate linear regression performs
