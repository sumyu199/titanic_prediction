import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# Load the passenger data
data = pd.read_csv('data.csv')
sample = pd.read_csv('sample.csv')
# Update sex column to numerical
data['Sex'] = data["Sex"].apply(lambda x:  1 if x == 'female' else 0)
sample['Sex'] = sample["Sex"].apply(lambda x:  1 if x == 'female' else 0)

# Fill the nan values in the age column
data['Age'] = data['Age'].fillna(value = np.mean(data['Age']))
sample['Age'] = sample['Age'].fillna(value = np.mean(sample['Age']))

# Create a first class column
data['FirstClass'] = data['Pclass'].apply(lambda x: 1 if x == 1 else 0)
sample['FirstClass'] = sample['Pclass'].apply(lambda x: 1 if x == 1 else 0)


# Create a second class column
data['SecondClass'] = data['Pclass'].apply(lambda x: 1 if x == 2 else 0)
sample['SecondClass'] = sample['Pclass'].apply(lambda x: 1 if x == 2 else 0)

# update cabin column to numerical
data['Cabin'] = data['Cabin'].fillna(value = 0)
data['Cabin'] = data['Cabin'].apply(lambda x: 1 if x !=  0 else 0)

sample['Cabin'] = sample['Cabin'].fillna(value = 0)
sample['Cabin'] = sample['Cabin'].apply(lambda x: 1 if x !=  0 else 0)

data['S'] = data['Embarked'].apply(lambda x: 1 if x == 'S' else 0)
data['C'] = data['Embarked'].apply(lambda x: 1 if x == 'C' else 0)

sample['S'] = sample['Embarked'].apply(lambda x: 1 if x == 'S' else 0)
sample['C'] = sample['Embarked'].apply(lambda x: 1 if x == 'C' else 0)

# Select the desired features
#,'Cabin',"SibSp",'Parch','FirstClass','SecondClass'
features_data = data[['Sex','Age',"SibSp",'Parch','FirstClass','SecondClass']]
sample_data = sample[['Sex','Age',"SibSp",'Parch','FirstClass','SecondClass']]
survival_labels = data['Survived']

train_data,test_data,train_labels,test_labels = train_test_split(features_data,survival_labels,random_state=1)


forest = RandomForestClassifier(random_state = 1)
forest.fit(train_data,train_labels)
print('forestscore',forest.score(test_data,test_labels))
# Score the model on the train data
print(forest.feature_importances_)

score = []
for i in range(1,21):
 tree = DecisionTreeClassifier(random_state = 1,max_depth = i)
 tree.fit(train_data,train_labels)
 score.append(tree.score(test_data,test_labels))
print(max(score))
print(score.index(max(score)))




tree = DecisionTreeClassifier(random_state = 1,max_depth = 4)
tree.fit(train_data,train_labels)
print(tree.score(test_data,test_labels))



survival_test = pd.DataFrame()
survival_test['Survived'] = tree.predict(sample_data).tolist()



test_result = pd.merge(sample['PassengerId'],survival_test,how = 'left',left_index = True, right_index = True)
test_result.to_csv(r'/Users/ngsumyu/PycharmProjects/titanic/venv/lib/python3.8/test_prediction.csv',index=False,header=True)
