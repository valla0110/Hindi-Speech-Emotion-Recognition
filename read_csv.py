import csv
import operator

happy=[]
sad=[]
angry=[]
fear=[]
surprise=[]
dataset={}
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

for row in happy:
    
    dataset.append('happy':row)
print(dataset)
