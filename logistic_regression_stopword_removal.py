import os
import re
from nltk.corpus import stopwords
import numpy as np

def readFile(path):
    files = os.listdir(path)
    vocabulary = []
    d = {}
    for file in files:
        f = open(path+"/"+file,encoding = "ISO-8859-1")
        text = f.read()
        text = re.sub('[^a-zA-Z]', ' ', text)
        words_list = text.strip().split()
        d[file] = words_list
        vocabulary.extend(words_list)
    return vocabulary, d

def readStopWords(path, stop_words):
    files = os.listdir(path)
    vocabulary = []
    d = {}
    for file in files:
        f = open(path+"/"+file,encoding = "ISO-8859-1")
        text = f.read()
        text = re.sub('[^a-zA-Z]', ' ', text)
        words_list = text.strip().split()
        words_new = []
        for w in words_list:
            if w not in stop_words:
                words_new.append(w)
        d[file] = words_new
        vocabulary.extend(words_new)
    return vocabulary, d

def extractKeys(train_spam_list,train_ham_list):
    return list(set(train_spam_list)|set(train_ham_list))

def getClassLabels(num_spam_file, num_ham_file):
    assigned_class = []
    for i in range(num_spam_file):
        assigned_class.append(0)
    for j in range(num_ham_file):
        assigned_class.append(1)
    return assigned_class

def featureList(words, dic1):
    w = list(words)
    result = []
    for f in dic1:
        row = [0] * (len(w))
        for word in w:
            if word in dic1[f]:
                row[w.index(word)] = 1
        row.insert(0,1)
        result.append(row)
    return result

def mergeData(dataset1, dataset2):
    dic3 = dataset1.copy()
    dic3.update(dataset2)
    return dic3

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def trainLR(data_matrix,class_labels,Lambda):
    data_set = np.mat(data_matrix)
    label_set = np.mat(class_labels).transpose()
    m,n = np.shape(data_set)
    alpha = 0.1
    iterations = 100
    weights = np.zeros((n,1))
    for k in range(iterations):
        print("Iteration number -", k)
        h = sigmoid(data_set*weights)
        error = (label_set - h)
        weights = weights + alpha * data_set.transpose() * error - alpha * Lambda * weights
    return weights


def classify(weight,data,num_spam,num_ham):
    matrix = np.mat(data)
    wx = matrix * weight
    correct = 0
    spam = 0
    ham = 0
    total = num_spam + num_ham

    for i in range(num_spam):
        if wx[i][0] < 0.0:
            correct += 1
            spam += 1
    for j in range(num_spam+1,total):
        if wx[j][0] > 0.0:
            correct += 1
            ham +=1
    
    print("Number of Ham emails correctly classifed: " , ham)
    print("Number of Ham emails incorrectly classifed: " , len(test_ham_dict) - ham)
    print("Ham Accuracy: ", (ham / len(test_ham_dict)) * 100)
    print("Number of Spam emails correctly classifed: " , spam)
    print("Number of Spam emails incorrectly classifed: " , len(test_spam_dict) - spam)
    print("Spam Accuracy: ", (spam / len(test_spam_dict)) * 100)
    print("Total Accuracy: ", 1.0 * correct/total *100)
    return wx


if __name__ == "__main__":
    train_spam_path = r'train/spam'
    train_ham_path = r'train/ham'
    test_spam_path = r'test/spam'
    test_ham_path = r'test/ham'
    stop_words = set(stopwords.words('english'))
    
    train_spam_list, train_spam_dict = readStopWords(train_spam_path,stop_words)
    train_ham_list, train_ham_dict = readStopWords(train_ham_path,stop_words)
        
    test_spam_list, test_spam_dict = readFile(test_spam_path)
    test_ham_list, test_ham_dict = readFile(test_ham_path)
    
    words = extractKeys(train_spam_list,train_ham_list)
    train = mergeData(train_spam_dict,train_ham_dict)
    test = mergeData(test_spam_dict,test_ham_dict)

    num_train_spam = len(train_spam_dict)
    num_test_spam = len(test_spam_dict)
    num_train_ham= len(train_ham_dict)
    num_test_ham = len(test_ham_dict)

    train_labels = getClassLabels(num_train_spam,num_train_ham)

    train_data_list = featureList(words, train)
    test_data_list = featureList(words, test)
    
    Lambda = float(input("Enter Lambda value:"))
    
    print("Number of test Ham emails: " , len(test_ham_dict))
    print("Number of test Spam emails: " , len(test_spam_dict))
    weight = trainLR(train_data_list, train_labels, Lambda)
    test = classify(weight, test_data_list, num_test_spam, num_test_ham)
    
