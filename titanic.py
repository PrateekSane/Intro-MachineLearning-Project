import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.optimizers import SGD

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

train_test_data = [train, test]

title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Col': 4, 'Rev': 4, 'Dona': 4, 'Dr': 4, 'Major': 4, 'Mlle': 4,
             'Don': 4, 'Sir': 4, 'Ms': 4, 'Lady': 4, 'Capt': 4, 'Countess': 4, 'Jonkheer': 4, 'Mme': 4}
sex_map = {'male': 0, 'female': 0}
emb_map = {'S': 0, 'Q': 1, 'C': 2, 'J': 3}
cabin_map = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}

for dataset in train_test_data:
    # Name Standardization
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].map(title_map)
    dataset.drop('Name', axis=1, inplace=True)

    # Sex standardization
    dataset['Sex'] = dataset['Sex'].map(sex_map)

    # Port Standardization
    dataset['Embarked'].fillna('S', inplace=True)
    dataset['Embarked'] = dataset['Embarked'].map(emb_map)

    # Age Standardization
    dataset['Age'].fillna(dataset.groupby('Title')['Age'].transform('median'), inplace=True)
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 62), 'Age'] = 4

    # Fare standardization
    dataset['Fare'].fillna(dataset.groupby('Title')['Fare'].transform('median'), inplace=True)
    dataset.loc[dataset['Fare'] <= 17, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 100), 'Fare'] = 3

    # Cabin Standardization
    dataset['Cabin'] = dataset['Cabin'].str[:1].map(cabin_map)
    dataset['Cabin'].fillna(dataset.groupby('Pclass')['Cabin'].transform('median'), inplace=True)

    # Fam Standardization
    dataset["FamSize"] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['FamSize'] = dataset['FamSize'].map(family_mapping)
    dataset.drop(['Ticket', 'SibSp', 'Parch', 'PassengerId'], axis=1, inplace=True)


y_train = train.pop('Survived')
print(train.head())

#print(train.isna().sum())
train = np.asarray(train)
y_train = np.asarray(y_train)

model = keras.Sequential()
model.add(keras.layers.Dense(20, input_dim=train.shape[1], activation='relu'))
model.add(keras.layers.Dropout(.5))
model.add(keras.layers.Dense(1, input_dim=20, activation='sigmoid'))

sgd = SGD(lr=.0001)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(train, y_train, epochs=10000, batch_size=32)

pred = model.predict(test)
with open("kaggle.csv", "w") as f:
    f.write("PassengerId,Survived\n")
    for index, prediction in zip(range(len(pred)), pred):
        f.write("{0},{1}\n".format(index, prediction))
"""
"""