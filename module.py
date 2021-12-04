#from flask import session
import pickle
#import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
wine_dataset = pd.read_csv('winequality-red.csv')
def functionf(y_value):
    if y_value<=3:
        y_value=1
        return y_value
    elif y_value==4:
        y_value=2
        return y_value
    elif y_value==5:
        y_value=3
        return y_value
    elif y_value==6:
        y_value=4
        return y_value
    elif y_value>=7:
        y_value=5
        return y_value
    else :
        y_value=-1
        return y_value
Y=[]
for i in range(len(wine_dataset['quality'])):
    y_value=functionf(wine_dataset['quality'][i])
    Y.append(y_value)
Y=pd.Series(Y)
X = wine_dataset.drop('quality',axis=1)
X=X[['volatile acidity','chlorides','total sulfur dioxide','density','sulphates','alcohol']]
#Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=3)
model = RandomForestClassifier()
model.fit(X_train, Y_train)
pickle.dump(model,open('model1.pkl','wb'))
#my_file=pickle.load(open('model.pkl','rb'))

#prediction=my_file.predict(input_data_reshaped)
#session["prediction"]=prediction
#print(prediction)