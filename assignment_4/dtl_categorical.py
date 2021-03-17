import pandas as pd
import numpy as np
from graphviz import Graph
from pprint import pprint

train = pd.read_csv("TDT4171/assignment_4/train.csv")
test = pd.read_csv("TDT4171/assignment_4/test.csv")

'''
The dataset contains the following columns: 

Survival:       Did the passenger survive (0=No, 1=Yes)
Pclass:         Ticket class (1=1st, 2=2nd, 3=3rd)                              [ Categorical variable ]
Name:           The name of the passenger                                       [ Continuous variable  ]
Sex:            Sex                                                             [ Categorical variable ]  
Age:            Age in years                                                    [ Disregarded          ]
SibSp:          Number of siblings/spouses aboard the Titanic                   [ Continuous variable  ]
Parch:          Number of parents/children aboard the Titanic                   [ Continuous variable  ]
Ticket:         Ticket number                                                   [ Continuous variable  ]
Fare:           Passenger fare                                                  [ Continuous variable  ]
Cabin:          Cabin number                                                    [ Disregarded          ]
Embarked:       Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)  [ Categorical variable ] 
'''

continous_variables = ['Name', 'Age', 'Ticket','Fare', 'Cabin', 'SibSp', 'Parch']

train_cat = train.drop(continous_variables, axis = 1)

def decision_tree_learning(examples, attributes, goal, parent_examples):
    print(len(examples))
    print(attributes)
    if len(examples) == 0:
        print(examples)
        print(' length of examples equals zero ')
        leaf = plurity_value(parent_examples, goal)
        return leaf
    elif same_classification(examples, goal):
        print(' all examples same classification ')
        leaf = examples[goal].iloc[0]
        return leaf
    elif not attributes:
        print(' no attributes left to split on ')
        leaf = plurity_value(examples, goal)
        return leaf
    else:   
        d = {}
        information_gain = (importance(examples, attributes, goal))     #dictionary
        print('information_gain: ', information_gain)
        best = min(information_gain, key = information_gain.get)
        v = examples[best].unique()
        new_attributes = attributes.copy()
        new_attributes.remove(best)
        for vk in v:
            s = {}
            print('new attributes:', new_attributes)
            print('vk: ', vk)
            print('best: ', best )
            exs = examples.loc[examples[best] == vk]
            sub = decision_tree_learning(exs, new_attributes, goal, examples)
            s[vk] = sub
            d[str(best) + ' - ' + str(vk)] = s
    return d

def importance(examples, attributes, goal):
    ''' returns a dictionary with information gain for each random variable '''
    d = {}
    for attribute in attributes:
        example = examples.groupby(goal)[attribute].value_counts().unstack(fill_value = 0).stack()
        l = []
        for elem in example:
            l.append(elem)
        T = l[:len(l)//2]
        F = l[len(l)//2:]
        # Entrophy(s) = - P(0)log2P(0) - P(1)log2P(1)
        entrophy = sum(list((-(T[i] / (T[i] + F[i]))* np.log2(T[i] / (T[i] + F[i]))) -
                (F[i] / (T[i] + F[i]))* np.log2(F[i] / (T[i] + F[i]))
                for i in range(len(T)))) / len(T)
        d[attribute] = entrophy
    return d

''' demo importance function '''
attr = train_cat.columns.tolist()[1:]

#plurity_train = train[['Survived', 'Pclass']]

def plurity_value(examples, goal):
    ''' returns final prediction when no more examples left. chooses most common, selects first if equal '''
    most_common = examples.groupby(goal).size().idxmax()
    return most_common

#print(plurity_value(train_disc, attr, 'Survived'))

def same_classification(examples, goal):
    ''' returns True if all examples has same classification '''
    examples = examples[goal].to_numpy()
    return (examples[0] == examples).all()

#print(same_classification(train_disc, "Survived"))

def split_continuous(examples, goal, attributes):
    for attr in attributes:
        sort = train[[goal, attr]]
        sort = sort.sort_values(attr)
        splitpoints = []
        for spit in splitpoint:
            return # calc entropy
    return

''' setup decision tree learning '''
attr = train_cat.columns.tolist()[1:]
parent_default = train_cat.sample()

d = decision_tree_learning(train_cat, attr, "Survived", parent_default)

pprint(d)


''' desperate and retarded solution '''

def accuracy(train, test):
    train = train[["Survived", "Sex"]]
    test = test[["Survived", "Sex"]]
    number_of_predictions_train = len(train)
    number_of_predictions_test = len(test)
    train = train.groupby('Survived')['Sex'].value_counts().unstack(fill_value = 0).stack()
    test = test.groupby('Survived')['Sex'].value_counts().unstack(fill_value = 0).stack()
    tr = []
    for i in train:
        tr.append(i)
    correct_predictions_train = (tr[1] + tr[2])
    accuracy_train = correct_predictions_train / number_of_predictions_train
    te = []
    for i in test:
        te.append(i)
    correct_predictions_test = (te[1] + te[2])
    accuracy_test = correct_predictions_test / number_of_predictions_test
    print("Accuracy test: ", accuracy_test)
    print("Accuracy train: ", accuracy_train)

accuracy(train, test)

''' tree '''

tree = Graph()

tree.node('Sex', 'Sex')
tree.edge('Sex', 'Pclass-f', label = 'female')
tree.node('Pclass-f', 'Pclass')
tree.node('Embarked1', 'Embarked')
tree.node('Embarked2', 'Embarked')
tree.node('Embarked3', 'Embarked')
tree.edge('Pclass-f', 'Embarked1', label = '1')
tree.edge('Pclass-f', 'Embarked2', label = '2')
tree.edge('Pclass-f', 'Embarked3', label = '3')


tree.edge('Sex', 'Embarked-m', label = 'male')
tree.node('Embarked-m', 'Embarked')
tree.node('Pclass-C', 'Pclass')
tree.node('Pclass-Q', 'Pclass')
tree.node('Pclass-S', 'Pclass')
tree.edge('Embarked-m', 'Pclass-C', label = 'C')
tree.edge('Embarked-m', 'Pclass-Q', label = 'Q')
tree.edge('Embarked-m', 'Pclass-S', label = 'S')


tree.node('C1', '1', shape='box')
tree.edge('Embarked1', 'C1', label = 'C')
tree.node('Q1', '1', shape='box')
tree.edge('Embarked1', 'Q1', label = 'Q')
tree.node('S1', '1', shape='box')
tree.edge('Embarked1', 'S1', label = 'S')

tree.node('C2', '1', shape='box')
tree.edge('Embarked2', 'C2', label = 'C')
tree.node('Q2', '1', shape='box')
tree.edge('Embarked2', 'Q2', label = 'Q')
tree.node('S2', '1', shape='box')
tree.edge('Embarked2', 'S2', label = 'S')

tree.node('C3', '1', shape='box')
tree.edge('Embarked3', 'C3', label = 'C')
tree.node('Q3', '1', shape='box')
tree.edge('Embarked3', 'Q3', label = 'Q')
tree.node('S3', '1', shape='box')
tree.edge('Embarked3', 'S3', label = 'S')


tree.node('11', '0', shape='box')
tree.edge('Pclass-C', '11', label = '1')
tree.node('21', '0', shape='box')
tree.edge('Pclass-C', '21', label = '2')
tree.node('31', '0', shape='box')
tree.edge('Pclass-C', '31', label = '3')

tree.node('12', '0', shape='box')
tree.edge('Pclass-Q', '12', label = '1')
tree.node('22', '0', shape='box')
tree.edge('Pclass-Q', '22', label = '2')
tree.node('32', '0', shape='box')
tree.edge('Pclass-Q', '32', label = '3')

tree.node('13', '0', shape='box')
tree.edge('Pclass-S', '13', label = '1')
tree.node('23', '0', shape='box')
tree.edge('Pclass-S', '23', label = '2')
tree.node('33', '0', shape='box')
tree.edge('Pclass-S', '33', label = '3')

tree.render()