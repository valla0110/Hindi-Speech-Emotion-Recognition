import numpy as np
from math import sqrt
from collections import Counter
import csv
import operator

happy=[]
sad=[]
angry=[]
fear=[]
surprise=[]
dataset={}

def create_data_set():
    with open('train_data.csv') as file:
    reader = csv.reader(file)
    for eachline in reader:
        if eachline[5] == '1':
            eachline.pop()
            angry.append(eachline)
        if eachline[5] == '2':
            eachline.pop()
            happy.append(eachline)
        if eachline[5] == '3':
            eachline.pop()
            sad.append(eachline)
        if eachline[5] == '4':
            eachline.pop()
           fear.append(eachline)
        if eachline[5] == '5':
            eachline.pop()
            surprise.append(eachline)
    

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
        
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

#dataset = {'happy':[[2,1,1,1,0],[2,2,1,1,0],[3,1,0,1,1]], 'sad':[[0,1,1,1,0],[0,2,1,1,1],[1,4,1,1,0]]}
new_features = [3,1,1,1,1]


result = k_nearest_neighbors(dataset, new_features)
print(result)

