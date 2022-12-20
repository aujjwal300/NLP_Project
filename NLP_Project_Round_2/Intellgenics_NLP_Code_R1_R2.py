# -*- coding: utf-8 -*-
"""Book Processing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VX5pX5WLBfV82-dBCfizXjeiCCC_QjE1
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
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS #For creating Wordclouds

#Mounting into Google Drive
from google.colab import drive
drive.mount('/content/drive')

#Reading data from the book text file
book = open("/content/drive/MyDrive/NLP/BE Book.txt","r",encoding='utf-8', errors='ignore')
word_list = book.read().splitlines()

#Determining no. of words and characters in book
print(f"No. of Words: {len(word_tokenize(' '.join(word_list)))}")
print(f"No. of Characters: {len(' '.join(word_tokenize(' '.join(word_list))))}")

#Adding Space at end of the string so that when it is merged 
word_list = [i+" " for i in word_list if i!='']
print(f"No. of Lines: {len(word_list)}")

print("\nPrinting only first 40 lines\n")
word_list[:40]

word_list = word_list[451:-1087] #Removing the Preface, Indexes and Answers

#Removing Running Chapter Names from Text
for i in word_list:
    if re.search(r"(^|[0-9]+\s)[cC]hapter\s[0-9]+",i) != None:
        word_list.remove(i)

#Converting List to the text by joining them
word_txt = ""
word_txt = word_txt.join(word_list)


word_txt = re.sub(r"\\", " ", word_txt)
word_txt = re.sub(r"[^0-9\s]*[0-9][^0-9\s]*", "", word_txt)
word_txt = re.sub(r" +", " ", word_txt)
word_txt = re.sub(r"- ", "", word_txt)

word_txt[:2000]

#Converting Data to Lower Case and then cleaning it
clean_txt = word_txt.lower()
clean_txt = re.sub("(\n)|(\s)+"," ",clean_txt)   # Removing new line character and multiple spaces

clean_txt = re.sub('[^(a-zA-Z)\s]', ' ', clean_txt)
my_punct = string.punctuation                    
punct_pattern = re.compile("[" + re.escape("".join(my_punct)) + "]")  
clean_txt = re.sub(punct_pattern, " ", clean_txt)                       # Removing all punctuation marks
clean_txt = re.sub("(\s)*[0-9]+(\s)+"," ",clean_txt)                    # Removing all the numbers
clean_txt = re.sub("(\s)[a-z](\s)"," ",clean_txt)                       # Removing single alphabetic characters
clean_txt = re.sub("(\s)+"," ",clean_txt)                               # Removing multiple spaces

clean_txt

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
df

#Determining the frequency based on Word length
len_df = df.groupby(['Length_Tokens'],as_index=False).sum()
len_df

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
len_df_a

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
df_a

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
tagged[:50]

dic_tag = dict(nltk.FreqDist([tag for (word,tag) in tagged]))
tag_df = pd.DataFrame(list(dic_tag.items()))
tag_df.rename(columns = {0:'Tags', 1:'Frequency'}, inplace = True)
tag_df = tag_df.sort_values('Frequency',ascending=False)
tag_df

plt.figure(figsize=(18,6))
plt.title("Line Plot for POS Tags and their frequency")
plt.plot(tag_df['Tags'],tag_df['Frequency'])
plt.xticks(rotation=90)
plt.xlabel("Tags")
plt.ylabel("Frequency")
plt.grid()
plt.show()

"""# **Project Round-2**


---


"""

from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('omw-1.4')

noun_count = 0
nouns={}
verb_count = 0
verbs={}

for i in tokens:
  try:
    tag = wordnet.synsets(i)[0]
    if tag.pos() == 'n':
      noun_count += 1
      lex_n = tag.lexname()
      lex_n = lex_n[5:]
      if lex_n not in nouns.keys():
        nouns[lex_n] = [i]
      else:
        nouns[lex_n].append(i)
    elif tag.pos() == 'v':
      verb_count += 1
      lex_v = tag.lexname()
      lex_v = lex_v[5:]
      if lex_v not in verbs.keys():
        verbs[lex_v] = [i]
      else:
        verbs[lex_v].append(i)
  except:
    temp = 1

noun_count, verb_count

nouns

verbs

noun_cat_freq = pd.DataFrame({k:len(v) for k,v in nouns.items()}.items())
verb_cat_freq = pd.DataFrame({k:len(v) for k,v in verbs.items()}.items())


noun_cat_freq.rename(columns = {0:'Categories', 1:'Frequency'}, inplace = True)
noun_cat_freq = noun_cat_freq.sort_values('Frequency',ascending=False)

verb_cat_freq.rename(columns = {0:'Categories', 1:'Frequency'}, inplace = True)
verb_cat_freq = verb_cat_freq.sort_values('Frequency',ascending=False)

# noun_cat_freq
verb_cat_freq

plt.figure(figsize=(20,10))
sns.barplot(x= 'Categories',y='Frequency', data=noun_cat_freq)
# sns.barplot(x= 'Categories',y='Frequency', data=verb_cat_freq)
plt.show()

import spacy
from spacy import displacy
from collections import Counter
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
nlp = spacy.load('en_core_web_sm')

ner = nlp.get_pipe("ner")
ner.add_label("PER")
ner.add_label("ORG")
ner.add_label("LOC")
ner.add_label("GPE")
ner.add_label("FAC")
ner.add_label("VEH")

book_text = nlp(clean_txt[:10000])
entity = [X.text for X in book_text.ents]
y_pred = [X.label_ for X in book_text.ents]

y_pred

print([(X, X.ent_iob_, X.ent_type_)for X in book_text][900:1010])

labels = [X.label_ for X in book_text.ents]
Counter(labels)

displacy.render((book_text), jupyter = True, style = 'ent')

# # Manually label the entities
# y_true = []
# for x in entity:
#     entity_label = input(f"Label for entity '{x}': ")
#     y_true.append(entity_label)

# print(y_true)

y_true = ['PRODUCT',
          'CARDINAL',
          'CARDINAL',
          'CARDINAL',
          'CARDINAL',
          'CARDINAL',
          'DATE',
          'CARDINAL',
          'ORDINAL',
          'ORDINAL',
          'CARDINAL',
          '',
          'CARDINAL',
          'CARDINAL',
          'CARDINAL',
          'CARDINAL']

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")

print(f"Accuracy: {accuracy:.3f}")
print(f"F1 Score: {f1:.3f}")

"""## Part-3"""

#Considering only the a small set

def bag_of_words(entity):
  neigh = []
  try:
      for i in range(-3,0):
        neigh.append(entity[0].nbor(i))  # text of token before the entity
  except:
      pass
  try:
      for i in range(1,4):
        neigh.append(entity[-1].nbor(i))  # text of token after the entity
  except:
      pass
  return neigh

table = []
for i in book_text.ents:
  for j in book_text.ents:
    if i.start < j.start:
      row = []
      row.append(i.text)
      row.append(j.text)
      row.append(i.label_)
      row.append(j.label_)
      row.append(bag_of_words(i))
      row.append(bag_of_words(j))
      table.append(row)

table = pd.DataFrame(table,columns=['Entity1','Entity2','Label_E1','Label_E2','BOW_E1','BOW_E2'])
table

dict_feat = {}
for ent1 in book_text.ents:
    for ent2 in book_text.ents:
      ent1_text = ent1.text
      ent2_text = ent2.text
      if (ent1_text,ent2_text) not in dict_feat.keys():
        dict_feat[(ent1_text,ent2_text)] = []
                  
      if ent1.start < ent2.start:
        # Extract the dependency parse tree of the sentence containing the entities
        sentence = book_text[ent1.sent.start:ent2.sent.end]
        dependencies = []
        for token in sentence:
          dependencies.append({
              "text": token.text,
              "dep": token.dep_,
              "head": token.head.text,
              "pos": token.pos_
          })
        dependency_tree = {"root": dependencies}

        # Extract the contextual features
        context = book_text[ent1.end:ent2.start]
        context_words = [token.text for token in context]
        context_length = len(context_words)

        # Extract the entity distances
        ent1_index = ent1[0].i
        ent2_index = ent2[0].i
        entity_distance = abs(ent1_index - ent2_index)

        # Extract the part-of-speech tags and dependency labels
        pos_tags = []
        dependency_labels = []
        for token in context:
          pos_tags.append(token.pos_)
          dependency_labels.append(token.dep_)

        dict_feat[(ent1_text,ent2_text)].append(dependency_tree)
        dict_feat[(ent1_text,ent2_text)].append(context_words)
        dict_feat[(ent1_text,ent2_text)].append(context_length)
        dict_feat[(ent1_text,ent2_text)].append(entity_distance)
        dict_feat[(ent1_text,ent2_text)].append(pos_tags)
        dict_feat[(ent1_text,ent2_text)].append(dependency_labels)

table['dependency_tree'] = [dict_feat[(e1,e2)][0] for e1,e2 in zip(table['Entity1'],table['Entity2']) if dict_feat[(e1,e2)] != []]
table['context_words'] = [dict_feat[(e1,e2)][1] for e1,e2 in zip(table['Entity1'],table['Entity2']) if dict_feat[(e1,e2)] != []]
table['context_length'] = [dict_feat[(e1,e2)][2] for e1,e2 in zip(table['Entity1'],table['Entity2']) if dict_feat[(e1,e2)] != []]
table['entity_distance'] = [dict_feat[(e1,e2)][3] for e1,e2 in zip(table['Entity1'],table['Entity2']) if dict_feat[(e1,e2)] != []]
table['pos_tags'] = [dict_feat[(e1,e2)][4] for e1,e2 in zip(table['Entity1'],table['Entity2']) if dict_feat[(e1,e2)] != []]
table['dependency_labels'] = [dict_feat[(e1,e2)][5] for e1,e2 in zip(table['Entity1'],table['Entity2']) if dict_feat[(e1,e2)] != []]
table