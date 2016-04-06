# Solution to the predict the missing grade
# problem on HackerRank ML track

import simplejson as json
import numpy as np
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble     import RandomForestClassifier

subjects = ['English', 'Physics', 'Chemistry', 'ComputerScience', 'Biology', \
            'PhysicalEducation', 'Economics', 'Accountancy', \
            'BusinessStudies']

subjects_model1 = ['English', 'Physics', 'Chemistry', 'ComputerScience']
subjects_model2 = ['English', 'Physics', 'Chemistry', 'PhysicalEducation']
subjects_model3 = ['English', 'Physics', 'Chemistry', 'Economics']

subject_models = { 1 : subjects_model1,
                   2 : subjects_model2,
                   3 : subjects_model3
}


def read_dataset(filename):
    data_list = []
    with open(filename, "r") as data_file:
        for record in data_file.readlines():
            data_record = json.loads(record[:-1])
            data_list.append(data_record)
    return data_list[1:]

def form_dataset(dataset, model_id):
    X,Y = [],[]
    model_fields = subject_models[model_id]
    
    for _id, record in enumerate(dataset):
        try:
            
            l = [record[f] for f in model_fields]
            X.append(l)
            Y.append(record.get('Mathematics', 0))
        except KeyError, e:
            pass
    return np.array(X),np.array(Y)


def train(X,Y):
    model = RandomForestClassifier()
    model.fit(X,Y)
    return model

    

    
