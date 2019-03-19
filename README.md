# Introduction 
This repository is for the development of user's talk classification by machine learning.

# Getting Started
Instllation: tbd

# Evaluation
tbd

# TODO

## Data Collection
[地域]
[出身]
[恋愛]
[食べ物] Done  
[映画] Done  
[テレビ] Done  
[スポーツ] Done  
[芸能人]
[本マンガアニメ] Done  
[人間関係] Done  
[家族]
[仕事]

## LDA model
1. import csv data as pandas
2. format as numpy array [string data, label]
3. segmentate string data by mecab
4. extract 4 nouns
5. make one-hot vector
6. make corpus
7. make dictionary
8. make LDA model

## CNN model
1. import csv data as pandas
2. format as numpy array [string data, label]
3. segmentate string data by mecab
4. extract 4 nouns
5. make one-hot vector
6. make CNN model by keras

## RNN model
1. import csv data as pandas
2. format as numpy array [string data, label]
3. segmentate string data by mecab
4. wash data (discarding symbol, mark, non-independent words)
5. make one-hot vector
6. make LSTM model by keras