import pandas as pd
import numpy as np
from graphviz import Graph
import pydot


'''
Implementation failed while trying to assign Graphviz nodes during recursion.

dtl_categorical.py has an alternative, but rather ugly, solution using a dictionary :/

'''

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

train_disc = train.drop(continous_variables, axis = 1)
#print(train_disc)
#print(train)

def decision_tree_learning(examples, attributes, goal, parent_examples, level = 0):
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
        information_gain = (importance(examples, attributes, goal))     #dictionary
        print(information_gain)
        best = min(information_gain, key = information_gain.get)
        tree = Graph()
        tree.node(str(id(best)), label = str(best))
        v = examples[best].unique()
        new_attributes = attributes.copy()
        new_attributes.remove(best)
        for vk in v:
            print('new:', new_attributes)
            print('vk: ', vk)
            print('best: ', best )
            exs = examples.loc[examples[best] == vk]
            sub = decision_tree_learning(exs, new_attributes, goal, examples, level)
            level += 1
            if not isinstance(sub, Graph):
                tree.edge(str(id(best)), str(sub) + ' - ' +  str(vk), label= str(vk))
            else:
                tree.subgraph(sub)
                tree.edge(str(id(best)), str(best) + str(level), label = str(vk))
                print(sub)
    return tree

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
attr = train_disc.columns.tolist()[1:]

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

parent_default = train_disc.sample()

d = decision_tree_learning(train_disc, attr, "Survived", parent_default)

d.render()
