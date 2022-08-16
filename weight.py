import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')
data2 = pd.get_dummies(data['Gender'])
data.drop('Gender',axis=1,inplace=True)
data = pd.concat([data,data2],axis=1)

y=data['Index']
x=data.drop(['Index'],axis=1)

scaler = StandardScaler()
x= scaler.fit_transform(x)
x=pd.DataFrame(x)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=101)

param_grid = {'n_estimators':[100,200,300,400,500,600,700,800,1000]}
grid_cv = GridSearchCV(RandomForestClassifier(random_state=101),param_grid,verbose=3)

grid_cv.fit(xtrain,ytrain)

pred = grid_cv.predict(xtest)

print('Acuuracy is:',accuracy_score(ytest,pred)*100)

pickle.dump(scaler,open('scaling_model','wb'))
pickle.dump(grid_cv,open('grid_model','wb'))