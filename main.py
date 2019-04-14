import pandas as pd
import pyodbc
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt


#Server and Database used in the MSSQL server is imputed in this string
conn_string= (
    r'DRIVER={SQL Server};'
    r'SERVER=DESKTOP-SHDVFGO\SQLEXPRESS;'
    r'DATABASE=titanic;'
    r'Trusted_Connection=yes;'
)

#Connection with previous string is established
conn=pyodbc.connect(conn_string)

#Performing a query that will be imported as a dataframe here
query='''SELECT * FROM dbo.train'''

#Reading the query and creating a resulting dataframe
df = pd.read_sql_query(query,conn)

df.head()

df.info()


query_total= '''
SELECT PassengerId,Pclass,Sex,Age,SibSp,Parch,Ticket,Fare,Embarked FROM dbo.train 
WHERE Age IS NOT NULL
UNION
SELECT PassengerId,Pclass,Sex,Age,SibSp,Parch,Ticket,Fare,Embarked FROM dbo.test
WHERE Age IS NOT NULL
'''
query_test='''
SELECT PassengerId,Pclass,Sex,Age,SibSp,Parch,Ticket,Fare,Embarked FROM dbo.test
'''
queryz='''SELECT PassengerId,Pclass,Sex,Age,SibSp,Parch,Ticket,Fare,Embarked FROM dbo.test
WHERE Age IS NULL
'''
 
query2 ='''
SELECT * FROM dbo.train
WHERE Age IS NOT NULL
'''


query3='''
SELECT * FROM dbo.train
WHERE Age IS NULL
'''
dfv= pd.read_sql_query(query_total,conn)
df_age = pd.read_sql_query(query2,conn)
dfnump=pd.read_sql_query(query_test,conn)

df_age['Embarked']=df_age['Embarked'].fillna('S')
dfv['Embarked']=dfv['Embarked'].fillna('S')
dfv['Fare']=dfv['Fare'].fillna(10)

#CLEANING AND USING TRAIN DATA WITH SURVIVED AS AN OUTCOME TO CREATE A MODEL THAT CAN PREDICT AGE VALUES FOR NULL VALUES IN THE TRAINING DATA

X_age = df_age[['Survived','Pclass','Sex','SibSp','Parch','Fare','Embarked']].values
y_age = df_age['Age'].values

#Adding a label encoder for Gender and embarked
from sklearn.preprocessing import LabelEncoder
labelencoder1=LabelEncoder()
X_age[:,2]=labelencoder1.fit_transform(X_age[:,2])
labelencoder2=LabelEncoder()
X_age[:,6]=labelencoder2.fit_transform(X_age[:,6])



#Converting elements of the matrix into floats
X_age=X_age.astype(float)


#

#Doing the same thing for second matrix which has age values missing
df_age2 = pd.read_sql_query(query3,conn)
X_age2 = df_age2[['Survived','Pclass','Sex','SibSp','Parch','Fare','Embarked']].values
labelencoder3=LabelEncoder()
X_age2[:,2]=labelencoder3.fit_transform(X_age2[:,2])
labelencoder4=LabelEncoder()
X_age2[:,6]=labelencoder4.fit_transform(X_age2[:,6])


X_age2=X_age2.astype(float)


#Fitting a random forest regressor of the values with age values specified, this is to create a classifier that predicts the values for the instances without ages
classifier3= RandomForestRegressor(n_estimators= 100)

classifier3.fit(X_age,y_age)


#Predictor being made to classify the values
ypred=classifier3.predict(X_age2)

#Rounding the age values 
np.around(ypred,out=ypred)

#Filling the dataframe with ages missing
df_age2['Age']=ypred

#Create a seperate dataframe which consists of the values of missing ages to the corresponding passenger IDs
dfx=df_age2[['PassengerId','Age']]

#Merging the dataframe with the missing age values with the dataframe that has a prediction for the missing age values
ypred.tolist()
result = pd.merge(df,dfx,how='left', on='PassengerId')


#Rename the age columns to have the same name (as when merged automatically there will be 2 age columns created with different names)
result.columns = result.columns.str.replace('.*?Age.*?', 'Age')
result.rename(index=str, columns={"Age_x": "Age", "Age_y": "Age"},inplace=True)
[]


#Stacking and unstacking the df as a trick to create the resulting dataframe 'result' which has a completed column with all of the age values
s = result.stack()
result=s.unstack()

#Filling missing values of embarked as embark location as 'S' as this was the mode and there were only 2 missing values
result['Embarked']=result['Embarked'].fillna('S')

#Taking the most important variables out for the final dataframe
result_df = result[['Age','Pclass','Sex','SibSp','Parch','Fare','Embarked']]
X = result_df.values
y= result['Survived'].astype(float)

#USING THE COMBINED DATAFRAME OF TEST AND TRAIN WITHOUT SURVIVED VALUES TO PREDICT AGE VALUES FOR THE TEST DATAFRAME NULL VALUES FOR AGE


#Adding a label encoder for Gender and embarked

Xtest_age = dfv[['Pclass','Sex','SibSp','Parch','Fare','Embarked']].values
ytest_age = dfv['Age'].values




from sklearn.preprocessing import LabelEncoder
labelencoder1x=LabelEncoder()
Xtest_age[:,1]=labelencoder1x.fit_transform(Xtest_age[:,1])
labelencoder2x=LabelEncoder()
Xtest_age[:,5]=labelencoder2x.fit_transform(Xtest_age[:,5])



dftest_age2 = pd.read_sql_query(queryz,conn)
Xtest_age2 = dftest_age2[['Pclass','Sex','SibSp','Parch','Fare','Embarked']].values
labelencoder3=LabelEncoder()
Xtest_age2[:,1]=labelencoder3.fit_transform(Xtest_age2[:,1])
labelencoder4=LabelEncoder()
Xtest_age2[:,5]=labelencoder4.fit_transform(Xtest_age2[:,5])


Xtest_age2=Xtest_age2.astype(float)


Xtest_age=Xtest_age.astype(float)
classifier2= RandomForestRegressor(n_estimators= 100)

classifier2.fit(Xtest_age,ytest_age)

y2pred=classifier2.predict(Xtest_age2)

dfnine= pd.read_sql_query(queryz,conn)

#
#Rounding the age values 
np.around(y2pred,out=y2pred)

#Filling the dataframe with ages missing
dfnine['Age']=y2pred

#Create a seperate dataframe which consists of the values of missing ages to the corresponding passenger IDs
dfjaja=dfnine[['PassengerId','Age']]

#Merging the dataframe with the missing age values with the dataframe that has a prediction for the missing age values
y2pred.tolist()
result2 = pd.merge(dfnump,dfjaja,how='left', on='PassengerId')

#Rename the age columns to have the same name (as when merged automatically there will be 2 age columns created with different names)
result2.columns = result2.columns.str.replace('.*?Age.*?', 'Age')
result2.rename(index=str, columns={"Age_x": "Age", "Age_y": "Age"},inplace=True)
[]


#Stacking and unstacking the df as a trick to create the resulting dataframe 'result' which has a completed column with all of the age values
s = result2.stack()
result2=s.unstack()

#Taking the most important variables out for the final dataframe
result2_df = result2[['Age','Pclass','Sex','SibSp','Parch','Fare','Embarked']]

result2_df.fillna(10,inplace=True)
X2 = result2_df.values


#Preparing final fitting of the test dataframe with null values imputed so that we can do random forest regression
labelencoder5x=LabelEncoder()
X2[:,2]=labelencoder5x.fit_transform(X2[:,2])
labelencoder6x=LabelEncoder()
X2[:,6]=labelencoder6x.fit_transform(X2[:,6])
X2=X2.astype(float)






###

#Preparing final fitting of the train dataframe with null values imputed so we can do random forest regression
labelencoder5=LabelEncoder()
X[:,2]=labelencoder5.fit_transform(X[:,2])
labelencoder6=LabelEncoder()
X[:,6]=labelencoder6.fit_transform(X[:,6])
X=X.astype(float)


from sklearn.ensemble import RandomForestRegressor

regressor=RandomForestRegressor()

regressor.fit(X,y)



#See correlation between each of the variables
import seaborn as sns
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)




importances = regressor.feature_importances_
std = np.std([tree.feature_importances_ for tree in regressor.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]


names = list(result_df)
imp = [names[indices[i]] for i in range(len(indices))]

print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), imp)
plt.xlim([-1, X.shape[1]])
plt.show()

#All of the features seem to have importance attached to it, and since we are looking to predict interpretation of the variables is not important so we'll include all variables in the matrix for final analysis

final_predictions=regressor.predict(X2)
