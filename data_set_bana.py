from collections import Counter
from nltk.tokenize import word_tokenize
import glob
import nltk
from nltk.corpus import stopwords
import csv

path ='C:/Users/DELL PC/Desktop/New folder/SURPRISE_hindi.txt'
files=glob.glob(path)

fear=[]

stop_words=set(stopwords.words('hindi'))

with open(path, encoding='utf-8') as s:
        for line in s:
            fear.append(line.rstrip())

print(fear)
            
