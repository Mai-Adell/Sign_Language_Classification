import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('./data.pickle', 'rb'))

# converting them from lists to numpy arrays as the classifier deals with the data as numpy arrays
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# then we will preprocess the data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# creating the model
model = RandomForestClassifier()

model.fit(X_train,y_train)

y_predict = model.predict(X_test)

score = accuracy_score(y_predict,y_test)

print('{}% of samples where classified correctly'.format(score*100))

# saving the model to be loaded later
f = open('model.p','wb')
pickle.dump({'model':model}, f)
f.close()
