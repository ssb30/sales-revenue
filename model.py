import numpy as np
import pandas as pd
from matplotlib import pyplot
import seaborn as sns
import pickle
advert = pd.read_csv('Advertising.csv')
advert.drop(['Unnamed: 0'],axis=1,inplace=True)
advert['interaction'] = advert['TV'] * advert['radio']
X=advert[['TV','radio','interaction']]
y=advert.sales
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(X_train,y_train)
y_pred= linreg.predict(X_test)
pickle.dump(linreg,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
