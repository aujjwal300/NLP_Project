# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 21:37:16 2022

@author: IntellGenics Group
"""
#Importing Libraries
import re #re to apply regular expression
import numpy as np #For mathematical Operations
import pandas as pd #Pandas to work with DataFrames
import string #For Performing String Operations(For Punctuations)

#For Natural Language Operations
import nltk 
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt #For Creating Plots
from wordcloud import WordCloud, STOPWORDS #For creating Wordclouds

#Reading data from the book text file
book = open("BE Book.txt","r",encoding='utf-8', errors='ignore')
word_list = book.read().splitlines()

#Determining no. of words and characters in book
print(f"No. of Words: {len(word_tokenize(' '.join(word_list)))}")
print(f"No. of Characters: {len(' '.join(word_tokenize(' '.join(word_list))))}")

#Adding Space at end of the string
word_list = [i+" " for i in word_list if i!='']

#Removing the Preface, Indexes and Answers
word_list = word_list[451:-1087]

#Removing Running Chapter Names from Text
for i in word_list:
    if re.search(r"[0-9]+\s[c|C]hapter\s[0-9]+",i) != None:
        word_list.remove(i)

#Converting List to the text by joining them
word_txt = ""
word_txt = word_txt.join(word_list)

#Converting Data to Lower Case and then cleaning it
clean_txt = word_txt.lower()
clean_txt = re.sub("(\n)|(\s)+"," ",clean_txt)   # Removing new line character and multiple spaces
clean_txt = re.sub("-\s","",clean_txt)           # Connecting a word which was separated by connector and a space (Eg-> Tech- nology)

clean_txt = re.sub('[^(a-zA-Z)\s]', ' ', clean_txt)
my_punct = string.punctuation                    
punct_pattern = re.compile("[" + re.escape("".join(my_punct)) + "]")  
clean_txt = re.sub(punct_pattern, " ", clean_txt)       # Removing all punctuation marks
clean_txt = re.sub("(\s)*[0-9]+(\s)+"," ",clean_txt)    # Removing all the numbers
clean_txt = re.sub("  ", " ", clean_txt)                # Replacing double spaces characters to single space
clean_txt = re.sub("(\s)[a-z](\s)"," ",clean_txt)


#Tokenizing the Text
tokens = word_tokenize(clean_txt)

#Determining the frequency of Tokens
dic = {}
for i in tokens:
    if i not in dic.keys():
        dic[i] = 1
    else:
        dic[i] += 1
        
#Forming DataFrame contaning the Tokens, their count and their length
df = pd.DataFrame(list(dic.items()))
df.rename(columns = {0:'Tokens', 1:'Frequency'}, inplace = True)
df["Length_Tokens"] = [len(i) for i in df['Tokens']]
#Sorting DataFrame in order of the Length
df = df.sort_values('Frequency',ascending = False)

#Determining the frequency based on Word length
len_df = df.groupby(['Length_Tokens'],as_index=False).sum()

#Plot between Word Length and the frequency of tokens of that length before removing StopWords
plt.figure(figsize=(18,6))
plt.title("Line Plot between Word Length and the frequency of tokens of that length before removing StopWords")
plt.plot(len_df['Length_Tokens'],len_df["Frequency"])
plt.xticks(np.linspace(0,21,22))
plt.xlabel("Word Length")
plt.ylabel("Frequency")
plt.grid()
plt.show()

#Plot between Tokens and their frequency before removing stopWords
plt.figure(figsize=(18,6))
plt.title("Line Plot between Tokens and their frequency before removing stopWords")
plt.plot(df['Tokens'].iloc[:30],df['Frequency'].iloc[:30])
plt.xticks(rotation=90)
plt.xlabel("Tokens")
plt.ylabel("Frequency")
plt.grid()
plt.show()

#Creating WordCloud before Removing StopWords
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white', stopwords={},
                min_font_size = 10).generate(clean_txt)
                
plt.figure(figsize = (10, 10), facecolor = None)
plt.title("Word Cloud for Tokens before removing stopWords\n")
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

#Adding some of the irrelevant words to stopwords
Words_to_add = ["figure","fig","example", "one", "two", "three", "four", "shown", "using", "table", "thus", "section", "first", "use", "may", "eight", "must", "used", 
                "b", "m", "x", "z", "will", "equal"]
for i in Words_to_add:
  STOPWORDS.add(i)
  
#Removing StopWords
tokens = [w for w in tokens if w not in STOPWORDS]
stop_words_removed_txt = " ".join(tokens)

tok_len = [len(i) for i in tokens]
dic = dict(nltk.FreqDist(tok_len))
len_df_a = pd.DataFrame(list(dic.items()))
len_df_a.rename(columns = {0:'Length_Tokens', 1:'Frequency'}, inplace = True)
len_df_a = len_df_a.sort_values('Length_Tokens')

#Plot between Word Length and the frequency of tokens of that length after removing StopWords
plt.figure(figsize=(18,6))
plt.title("Line Plot between Word Length and the frequency of tokens of that length after removing StopWords")
plt.plot(len_df_a['Length_Tokens'],len_df_a['Frequency'])
plt.xticks(np.linspace(0,21,22))
plt.xlabel("Word Length")
plt.ylabel("Frequency")
plt.grid()
plt.show()

dic_a = dict(nltk.FreqDist(tokens))
df_a = pd.DataFrame(list(dic_a.items()))
df_a.rename(columns = {0:'Tokens', 1:'Frequency'}, inplace = True)
df_a = df_a.sort_values('Frequency',ascending=False)


plt.figure(figsize=(18,6))
plt.title("Line Plot between Tokens and their frequency after removing stopWords")
plt.plot(df_a['Tokens'].iloc[:30],df_a['Frequency'].iloc[:30])
plt.xticks(rotation=90)
plt.xlabel("Tokens")
plt.ylabel("Frequency")
plt.grid()
plt.show()

#Creating WordCloud after Removing StopWords
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white', stopwords={},
                min_font_size = 10).generate(stop_words_removed_txt)

plt.figure(figsize = (10, 10), facecolor = None)
plt.title("Word Cloud for Tokens after removing stopWords\n")
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0) 
plt.show()

#Performing POS Tagging using Penn Treebank Tagset
tagged = nltk.pos_tag(tokens)

dic_tag = dict(nltk.FreqDist([tag for (word,tag) in tagged]))
tag_df = pd.DataFrame(list(dic_tag.items()))
tag_df.rename(columns = {0:'Tags', 1:'Frequency'}, inplace = True)
tag_df = tag_df.sort_values('Frequency',ascending=False)

plt.figure(figsize=(18,6))
plt.title("Line Plot for POS Tags and their frequency")
plt.plot(tag_df['Tags'],tag_df['Frequency'])
plt.xticks(rotation=90)
plt.xlabel("Tags")
plt.ylabel("Frequency")
plt.grid()
plt.show()