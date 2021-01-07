import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

from regressors import stats

import seaborn as sns

# from keras.models import Sequential
# from keras.layers import Dense
# from keras import metrics

from matplotlib import pyplot as plt

df = pd.read_csv('turkey_car_market2.csv')

print('log: {}, no log: {}'.format(np.log(df['price']).skew(), (df['price']).skew()))
print('log: {}, no log: {}'.format(np.log(df['price']).kurt(), (df['price']).kurt()))

print(df.info())
print(df.columns)

# Parse date to datetime and create class based on grade

df['date'] = df.apply(lambda data: datetime.strptime(data['date'], '%Y%m%dT%H%M%S'), axis=1)
df['gradeClass'] = df.apply(lambda data: 1 if data['grade'] > 7 else 0, axis=1)

# print('\n', '='*50 + '\n')
# print('class 1: {}, class 2: {}\n'.format(sum(df['gradeClass']==0), sum(df['gradeClass']==1)))
# print('='*50 + '\n')
print(df.isna().sum())
for column in df.columns:
    if df[column].isna().sum() > 0: 
        df[column].fillna(df[column].median(), inplace=True)

    
# for column in df.columns:
#     print('{} ({}): \n{}\n'.format(column, type(df[column][10]), df[column][10]))

print(df.isna().sum())
print(df.columns[4:].drop(['zipcode', 'lat']))

# linear Model #
'''
linear_model = LinearRegression() 

x = df[df.columns[4:].drop(['zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'gradeClass', 'sqft_living', 'sqft_lot', 'yr_renovated'])] 
y = np.log(df['price']) 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

linear_model.fit(x_train, y_train)

y_pred = linear_model.predict(x_test)

metrics.r2_score(y_test, y_pred)

stats.summary(linear_model, x_train, y_train, x_train.columns)

plt.scatter(y_pred, y_test)
plt.plot(    
    np.arange(min(y_test.values), max(y_test.values)), 
    np.arange(min(y_test.values), max(y_test.values)), 
    c='red',
    linewidth=2
    )
plt.title('Predicted values vs Real values')
plt.xlabel('Predicted Values')
plt.ylabel('Real Values')
plt.show()
'''
# Logistic Model #
'''
Make_label = LabelEncoder()

x = df[df.columns[3:].drop(['zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'gradeClass', 'sqft_living', 'sqft_lot', 'yr_renovated', 'grade'])] 
x['price'] = np.log(x['price'])
print('vars', x.columns)
y = Make_label.fit_transform(df['gradeClass'])

print(set(df['grade']))
print('Total for class 1 (grade class = 0): {} ---- Total for class 2 (grade class = 1): {}'.format(sum(y==0), sum(y==1)))

x = StandardScaler().fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

logit_model = LogisticRegression()

logit_model.fit(x_train, y_train)

y_pred = logit_model.predict(x_test)

def accuracy(y_pred, y_test):
    acc = sum(y_pred == y_test)/len(y_pred)
    return acc

print(len(y_pred), 'Logit Accuracy: {}'.format(accuracy(y_pred, y_test)))

print(metrics.confusion_matrix(y_test, y_pred))
print(np.sort(logit_model.predict_proba(x_test)[:,1], axis=0))

titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]

for title, normalize in titles_options:
    disp = metrics.plot_confusion_matrix(logit_model, x_test, y_test,
                                 display_labels=['Low Grade', 'High Grade'],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
'''
# KNN
'''
Make_label = LabelEncoder()

x = df[df.columns[3:].drop(['zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'gradeClass', 'sqft_living', 'sqft_lot', 'yr_renovated', 'grade'])] 
x['price'] = np.log(x['price'])
# x = StandardScaler().fit_transform(x)

y = Make_label.fit_transform(df['grade'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print('Number of different classes: {}'.format(len(set(y))))

knn = KNeighborsClassifier(n_neighbors=9)

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

print('Accuracy: {}'.format(sum(y_pred == y_test)/len(y_test)))

# for i in range(100):
#     knn = KNeighborsClassifier(n_neighbors=i+1)

#     knn.fit(x_train, y_train)

#     y_pred = knn.predict(x_test)

#     print('Accuracy with {} clusters: {}'.format(i, sum(y_pred == y_test)/len(y_test)))

print(list(set(y_pred)))

plt.scatter(np.exp(x_test['price']), y_test, c=y_pred,
            s=50, cmap='viridis')
plt.title('Price log by cluster')
plt.xlabel('Price Log')
plt.ylabel('Grade class')
plt.show()

titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]

for title, normalize in titles_options:
    disp = metrics.plot_confusion_matrix(knn, x_test, y_test,
                                #  display_labels=['Low Grade', 'High Grade'],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
'''
# Kmeans
'''
Make_label = LabelEncoder()

x = df[df.columns[4:].drop(['zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'gradeClass', 'sqft_living', 'sqft_lot', 'yr_renovated', 'grade'])] 
x = StandardScaler().fit_transform(x)

y = Make_label.fit_transform(df['grade'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print('Number of different classes: {}'.format(len(set(y))))

kmeans = KMeans(n_clusters=11, random_state=0)
kmeans.fit(x_train, y_train)

y_pred = kmeans.predict(x_test)

print('Accuracy: {}'.format(sum(y_pred == y_test)/len(y_test)))
'''