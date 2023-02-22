#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import pandas as pd
import random
from nltk.stem.porter import *


#Total ham: 3672
#Total spam: 1500
ham_dir = './ham'
spam_dir = './spam'
all_dir = './all'

total_number = 3672+1500
#change training ratio

training_ratio=0.9

training_number = int(total_number * training_ratio)
test_number = total_number - training_number

training_index = random.sample(list(range(total_number)), training_number)
training_index = np.array(training_index)
print(training_index)

test_index = np.ones(total_number, dtype=bool)
test_index[training_index] = False;

word_set = set()

emails_all = []
labels_all = np.zeros(total_number)

count = 0
for filename in os.scandir(all_dir): 
    with open(filename, encoding='latin1') as f:
        if 'ham' in str(filename):
            labels_all[count] = 1
        lines = f.readlines()
        email_split = []
        for line in lines:
            line = line.lower()
            line_split = re.split(r'\s+', line)
            email_split = email_split + line_split

        #only add the words in the training to the dictionary
        if count in training_index:
            word_set.update(set(email_split))
            
        emails_all.append(email_split)
        count +=1 


#getting (training) priors for ham and spam
training_labels = labels_all[training_index]
test_labels = labels_all[test_index]
ham_prior = np.sum(training_labels)/training_number
spam_prior = 1 - ham_prior

#word stemming
stemmer = PorterStemmer()
word_stemmed = []
for word in word_set:
    word_stemmed.append(stemmer.stem(word))

#filtering out numbers and meaningless words
word_effective = []
meaningless_words = ['the','and','are','about']

for word in word_stemmed:
    if (word.isalpha()) & (len(word) > 2) & (word not in meaningless_words):
        word_effective.append(word)
print(len(word_effective))

#remove duplicates 
word_effective = set(word_effective)

#remove all words that only appeared once in all emails

word_count_all = {}

for word in word_effective:
        word_count_all[word]=0

for i in range(len(emails_all)):
    email = emails_all[i] 
    for word in email: 
        if word in word_count_all:
            word_count_all[word] = word_count_all[word] + 1 

word_all = list(word_count_all.keys())

for i in range(len(word_all)):
    word = word_all[i]
    if word_count_all[word] <= 1:
        del word_count_all[word]

final_dict =  list(word_count_all.keys())
final_dict_df = pd.DataFrame(final_dict)
word_number = len(final_dict)

print(word_number)
print(ham_prior, spam_prior)

###### embedding all data
word_number = len(final_dict)
all_data = np.zeros((word_number, total_number))

for i in range(total_number):
    email = emails_all[i]
    for word in email:
        if word in final_dict:
            all_data[final_dict.index(word),i] += 1;

all_data = np.row_stack((all_data, labels_all))
#the last row is the label

training_data = all_data[:, training_index]
test_data = all_data[:, test_index]
print(test_data.shape)

training_data_ham = training_data[:, training_data[-1,:]==1]
training_data_spam = training_data[:, training_data[-1,:]==0]

training_data_ham = np.delete(training_data_ham, -1, 0)
training_data_spam = np.delete(training_data_spam, -1, 0)

#k-nn vanilla version:
#first concatenate the test vector training_number times

knn_test_labels = np.zeros(test_number)
training_data_no_label = np.delete(training_data, -1, 0)

def k_nn(test_number, test_data, k, L):
    for i in range(test_number):
        test_data_no_label = np.delete(test_data, -1, 0)
        test_vec = test_data_no_label[:, i].reshape((word_number, 1))
        test_mat = np.broadcast_to(test_vec, (word_number, training_number))
        distance_mat = test_mat - training_data_no_label
        
        #select metric
        if L==2: #L2
            distances = np.linalg.norm(distance_mat, axis=0)
        elif L==1: #l1
            distances = np.sum(np.absolute(distance_mat),axis =0)
        elif L>2: #L_inf
            distances = np.amax(np.absolute(distance_mat), axis=0)
        
        #getting the index of the k smallest distances
        knn_idx = (np.argpartition(distances, k))[:k]
        
        knn_ham = np.sum(training_labels[knn_idx])
        knn_spam = knn_idx.shape[0] - knn_ham

        if (knn_ham > knn_spam):
            knn_test_labels[i] = 1
        else:
            knn_test_labels[i] = 0

    wrong_labels = np.nonzero(knn_test_labels - test_data[-1,:])
    error_rate = (wrong_labels[0].shape[0])/test_number
    print(1-error_rate)

#testing k-nn
k_nn(test_number, test_data, 1, 2)

def naive_bayes(test_number, test_data, ham_prior, spam_prior, alpha):
    
    bayes_test_labels = np.zeros(test_number)
    
    ham_word_count = np.sum(training_data_ham, axis = 1)
    ham_total = np.sum(ham_word_count)
    
    spam_word_count = np.sum(training_data_spam, axis = 1)
    spam_total = np.sum(spam_word_count)
    
    #additive smoothing
    ham_word_freq = (ham_word_count + alpha)/(ham_total + alpha*word_number)
    spam_word_freq = (spam_word_count + alpha)/(spam_total + alpha*word_number)

    for i in range(test_number):
        
        test_data_no_label = np.delete(test_data, -1, 0)
        test_vec = test_data_no_label[:,i] 
        
        spam_conditional = np.sum(test_vec * np.log(spam_word_freq))
        ham_conditional = np.sum(test_vec * np.log(ham_word_freq))
        
        bayes_test_labels[i] = np.argmax(np.array([spam_conditional + np.log(spam_prior), ham_conditional + np.log(ham_prior)]))
        
    wrong_labels = np.nonzero(bayes_test_labels - test_data[-1,:test_number])
    error_rate = (wrong_labels[0].shape[0])/test_number
    return (1-error_rate)

#testing naive bayes
print(naive_bayes(test_number, test_data, ham_prior, spam_prior, 1))





