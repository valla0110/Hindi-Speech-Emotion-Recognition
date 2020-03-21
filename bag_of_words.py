from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import  csv
from collections import Counter

print('hello')

vectorizer = CountVectorizer()
string1="hi nice to meet you"
string2="how you been"
string3="its nice looks good"
string4="you look good"
string5="you have been nice"
#string1="आपका स्वागत हैं एक मिनट स्वागत "
#string2="अरे दोस्त स्वागत "
#string3="आपका स्वागत हैं "
#string4="आप मेरे सबसे अच्छे दोस्त हैं"

text_list=[string1,string2,string3,string4,string5]

bag_of_words=vectorizer.fit(text_list)
bag_of_words=vectorizer.transform(text_list).toarray()

print(bag_of_words)

count_array=[]

for i in range(len(bag_of_words)):
    count=0
    for j in range(len(bag_of_words[0])):
        if bag_of_words[i][j]==1:
            count=count+1
    count_array.append(count)
            
print(count_array)           

print(vectorizer.vocabulary_)
print(vectorizer.get_feature_names())



"""with open("train_data.csv","w") as f:
    wr = csv.writer(f)
    wr.writerow(vectorizer.get_feature_names())
    wr.writerows(bag_of_words)"""
    
