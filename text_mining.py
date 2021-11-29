#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the glob library for Retrieving a list of all text files present in a folder
import glob

#The book files are contained in the folder Desertation/Books_data
data_folder = "Books_data/"

#List all the .txt files and sort them alphabetically
files = glob.glob(data_folder + "*.txt")

files.sort()
files


# In[2]:


# Import libraries
import re, os

# Initialize the object that will contain the texts and titles
books = []
titles = []

for p in files:
    # Open each file
    files = open(p, encoding='utf-8-sig')
    # Remove all non-alpha-numeric characters
    data = re.sub('[\W_]+', ' ', files.read())
    # Store the texts and titles of the books in two separate lists
    books.append(data)
    #Using os.path.basename() and replace() functions to remove the folder name and .txt extension from the file name.
    titles.append(os.path.basename(p).replace(".txt", ""))
    


# In[3]:


# Counting characters in each book
characters = [len(t) for t in books]
characters


# In[4]:


#counting the number of words in each book
words = [len(words.split()) for words in books]
words


# In[5]:


titles 


# In[6]:


authors = ["Roman Gelperin","Daniel H Pink", "Jenny Loveless","Ariely Dan", "Nick Winter" ]
authors


# In[11]:


# Creating a dictinary object to store index, titles, authors and book texts
Book_dict = {"Titles":titles, "authors": authors, "book_texts":books}


# In[12]:


import pandas as pd
# Creating a dataframe using dictory object
Books_dataFrame = pd.DataFrame(Book_dict)
Books_dataFrame


# In[10]:


# Books_dataFrame.to_csv(r'C:\Users\Sudheena Sona\Desktop\Desertation\my_books.csv', index = False)    


# In[13]:


#Bar plot
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.bar(titles, words, color = 'green', width = 0.5)
plt.xlabel("Book name", color = "blue")
plt.ylabel("word count", color = "blue")
plt.title("Number of words in each books", color="blue" )
plt.xticks(rotation=90, color= "red")
plt.yticks(color='red')
#plt.savefig("words.pdf")
plt.show()


# In[14]:


#Storing the texts of each books into sepearte variables for future reference
#Book_1 = Addiction procastination and laziness
book_1 = books[0]
# Book_2 = Drive The Surprising Truth about What Motivates Us
book_2 = books[1]
# Book_3 = Law of Attraction The Secret Power of The Universe 
book_3 = books[2]
# Book_4 = Payoff The Hidden Logic That Shapes Our Motivations
book_4 = books[3]
# Book_5 = The Motivation Hacker
book_5 = books[4]


# In[15]:


#Defining stop words
stop_words = ['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', 
"aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn',
"couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 
'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 
'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', 
"isn't", 'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', 
"mustn't", 'my','myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 
'other', 'our', 'ours', 'ourselves', 'out', 'over', 'yours', 'yourself', 'yourselves',"you'll", "you're", "you've", 'your',
'own', 're', 's', 'same', 'shan', "shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 
'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 
'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 
'we', 'were', 'weren', "weren't", 'what', 'when', 'where', 'which', 're', 's', 'same', 'shan', "shan't", 'she', 
"she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 
'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 
'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 
'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', 
"wouldn't", 'y', 'you', "you'd"]
       


# In[16]:


# Convert the texts to lower case 
lower_case = [letters.lower() for letters in books]
lower_case


# In[18]:


# Transform the text into tokens 
texts_tokens = [txt.split() for txt in lower_case]

# Remove tokens which are part of the list of stop words
books = [[word for word in txt if word not in stop_words] for txt in texts_tokens]


# 
#  Printing the first 15 tokens of each book
# 

# In[19]:



# Book 1
books[0][0:15]


# In[20]:


# Book 2
books[1][0:15]


# In[21]:


#Book 3
books[2][0:15]


# In[19]:


#Book 4
books[3][0:15]


# In[20]:


#Book 5
books[4][0:15]


# when we go throgh the texts of each book we can find some words listed below which seems irrelevant to our analysis 

# In[22]:


irrelevent_words = ["table","of",'contents', "introduction","conclusion","chapters","chapter", 
                    "one","two","three", "four", "five","six","seven", "eight","nine",
                    "ten","eleven","twelve","thirteen", "0","1","2","3","4","5","6","7","8","9","10","11","12","13"]


# In[23]:


# Remove irrelevent words
books = [[word for word in text if word not in irrelevent_words] for text in books]


# In[24]:


books[0:15]


# In[27]:


#Stemming
import  nltk
from nltk.stem import PorterStemmer
ps = PorterStemmer()


# In[25]:


# protection	protect	protective	protectively
#print(ps.stem("protecting"))


# In[28]:


#Generating stem for each words
stems = [[ps.stem(words) for words in text] for text in books]

stems


# In[29]:


stems[0][0:10]  


# In[24]:


stems[1][0:10]


# In[25]:


stems[2][0:10]


# In[30]:


stems[3][0:10]


# In[26]:


stems[4][0:10]


# #### Counting the number of words after pre-processing in each book

# In[30]:


# Before pre-processing the number of words in each books was the following
#number of words = [33048, 51310, 31692, 24618, 42594]

words_after_preprocessing = [len(words) for words in stems]
words_after_preprocessing


# #### Generating a bag of words model

# In[31]:


# Load the functions allowing to create and use dictionaries
from gensim import corpora

# Create a dictionary from the stemmed tokens
dict = corpora.Dictionary(stems)

print(dict)


# In[32]:


# Create a bag-of-words model for each book, using the previously generated dictionary
bag_of_words = [dict.doc2bow(text) for text in stems]


# Print the first five elements of the first book Bag of words model
bag_of_words[0][0:10]


# #### Counting the number of words in BOW model (unique words)

# In[33]:


bow_words = [len(words) for words in bag_of_words]
bow_words


# #### Most common words

# Generating dataframe for each book which represent the frequency of  words in each book

# In[34]:


# Book 1

# Import pandas to create and manipulate DataFrames
import pandas as pd

# Convert the Bag of words model for book 1 into a DataFrame
book_1_df = pd.DataFrame(bag_of_words[0])

# Add the column names to the DataFrame
book_1_df.columns = ["Index","Number of occurrences"]

# Add a column containing the token corresponding to the dictionary index
book_1_df["Book1_token"] = [dict[index] for index in book_1_df["Index"]]

# Sort the DataFrame by descending number of occurrences and print the first 7 values
book1_df_head = book_1_df.sort_values(by="Number of occurrences", ascending= False).head(7)
book1_df_head


# In[35]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.bar(book1_df_head["Book1_token"], book1_df_head["Number of occurrences"], color = 'brown', width = 0.5)
plt.xlabel("Stemmed words", color = "red")
plt.ylabel("Number of occurrences", color = "red")
plt.title("Top 7 Most common words of 'Addiction, Procrastination, and Laziness' ", color="red" )
plt.xticks(rotation=0, color= "blue")
plt.yticks(color='blue')
#plt.savefig("words.pdf")
plt.show()


# In[36]:


#book 2

# Convert the Bag of words model for book 2 into a DataFrame
book_2_df = pd.DataFrame(bag_of_words[1])

# Add the column names to the DataFrame
book_2_df.columns = ["Index","Number of occurrences"]

# Add a column containing the token corresponding to the dictionary index
book_2_df["Book2_token"] = [dict[index] for index in book_2_df["Index"]]

# Sort the DataFrame by descending number of occurrences and print the first 10 values
book2_df_head = book_2_df.sort_values(by="Number of occurrences", ascending=False).head(7)
book2_df_head


# In[37]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.bar(book2_df_head["Book2_token"], book2_df_head["Number of occurrences"], color = 'brown', width = 0.5)
plt.xlabel("Stemmed words", color = "red")
plt.ylabel("Number of occurrences", color = "red")
plt.title("Top 7 Most common words of 'Drive The Surprising Truth about What Motivates Us' ", color="red" )
plt.xticks(rotation=0, color= "blue")
plt.yticks(color='blue')
#plt.savefig("words.pdf")
plt.show()


# In[38]:


#book 3

# Convert the Bag of words model for book 3 into a DataFrame
book_3_df = pd.DataFrame(bag_of_words[2])

# Add the column names to the DataFrame
book_3_df.columns = ["Index","Number of occurrences"]

# Add a column containing the token corresponding to the dictionary index
book_3_df["Book3_token"] = [dict[index] for index in book_3_df["Index"]]

# Sort the DataFrame by descending number of occurrences and print the first 10 values
book3_df_head = book_3_df.sort_values(by="Number of occurrences", ascending=False).head(7)
book3_df_head


# In[39]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.bar(book3_df_head["Book3_token"], book3_df_head["Number of occurrences"], color = 'brown', width = 0.5)
plt.xlabel("Stemmed words", color = "red")
plt.ylabel("Number of occurrences", color = "red")
plt.title("Top 7 Most common words of 'The Law of Attraction, The Secret Power Of The Universe' ", color="red" )
plt.xticks(rotation=0, color= "blue")
plt.yticks(color='blue')
#plt.savefig("words.pdf")
plt.show()


# In[40]:


#book 4

# Convert the Bag of words model for book 4 into a DataFrame
book_4_df = pd.DataFrame(bag_of_words[3])

# Add the column names to the DataFrame
book_4_df.columns = ["Index","Number of occurrences"]

# Add a column containing the token corresponding to the dictionary index
book_4_df["Book4_token"] = [dict[index] for index in book_4_df["Index"]]

# Sort the DataFrame by descending number of occurrences and print the first 10 values
book4_df_head = book_4_df.sort_values(by="Number of occurrences", ascending=False).head(7)
book4_df_head


# In[41]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.bar(book4_df_head["Book4_token"], book4_df_head["Number of occurrences"], color = 'brown', width = 0.5)
plt.xlabel("Stemmed words", color = "red")
plt.ylabel("Number of occurrences", color = "red")
plt.title("Top 7 Most common words of 'Payoff The Hidden Logic That Shapes Our Motivations' ", color="red" )
plt.xticks(rotation=0, color= "blue")
plt.yticks(color='blue')
#plt.savefig("words.pdf")
plt.show()


# In[42]:


#Book 5

# Convert the Bag of words model for book 5 into a DataFrame
book_5_df = pd.DataFrame(bag_of_words[4])

# Add the column names to the DataFrame
book_5_df.columns = ["Index","Number of occurrences"]

# Add a column containing the token corresponding to the dictionary index
book_5_df["Book5_token"] = [dict[index] for index in book_5_df["Index"]]

# Sort the DataFrame by descending number of occurrences and print the first 10 values
book5_df_head = book_5_df.sort_values(by="Number of occurrences", ascending=False).head(7)
book5_df_head


# In[43]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.bar(book5_df_head["Book5_token"], book5_df_head["Number of occurrences"], color = 'brown', width = 0.5)
plt.xlabel("Stemmed words", color = "red")
plt.ylabel("Number of occurrences", color = "red")
plt.title("Top 7 Most common words of 'The Motivation Hacker ' ", color="red" )
plt.xticks(rotation=0, color= "blue")
plt.yticks(color='blue')
#plt.savefig("words.pdf")
plt.show()


# ### Building a term frequency - inverse document frequency model(tf-idf model)

# In[44]:


# Load the gensim functions that will allow us to generate tf-idf models
from gensim.models import TfidfModel

# Generate the tf-idf model
tf_model = TfidfModel(bag_of_words)


# #### Inspecting the tf_idf words of each book seperately

# ## Book1

# In[45]:


# Print the model for book 1
tf_model[bag_of_words[0]]


# In[46]:


# Convert the tf-idf model for book 1 into a DataFrame
df_tf_idf = pd.DataFrame(tf_model[bag_of_words[0]])

# Name the columns of the DataFrame id and score
df_tf_idf.columns=["id", "tf_idf_score"]

# Add the tokens corresponding to the numerical indices for better readability
df_tf_idf['Book1_words'] = [dict[i] for i in list(df_tf_idf["id"])]

# Sort the DataFrame by descending tf-idf score and print the first 7 rows.
book1_tfidf_head = df_tf_idf.sort_values(by="tf_idf_score", ascending=False).head(5)
book1_tfidf_head


# In[47]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.bar(book1_tfidf_head["Book1_words"], book1_tfidf_head["tf_idf_score"], color = 'purple', width = 0.5)
plt.xlabel("Book1 Words", color = "green")
plt.ylabel("TF-IDF Scores", color = "green")
plt.title("Top 5 most significant words of 'Addiction, Procrastination, and Laziness' ", color="red" )
plt.xticks(rotation=0, color= "blue")
plt.yticks(color='blue')
#plt.savefig("words.pdf")
plt.show()


# ### Book 2

# In[48]:


# Print the model for book 2
#tf_model[bag_of_words[1]]


# In[49]:


# Convert the tf-idf model for book 1 into a DataFrame
df_tf_idf = pd.DataFrame(tf_model[bag_of_words[1]])

# Name the columns of the DataFrame id and score
df_tf_idf.columns=["id", "tf_idf_score"]

# Add the tokens corresponding to the numerical indices for better readability
df_tf_idf['Book2_words'] = [dict[i] for i in list(df_tf_idf["id"])]

# Sort the DataFrame by descending tf-idf score and print the first 10 rows.
book2_tfidf_head = df_tf_idf.sort_values(by="tf_idf_score", ascending=False).head(5)
book2_tfidf_head


# In[50]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.bar(book2_tfidf_head["Book2_words"], book2_tfidf_head["tf_idf_score"], color = 'purple', width = 0.5)
plt.xlabel("Book2 Words", color = "green")
plt.ylabel("TF-IDF Scores", color = "green")
plt.title("Top 5 most significant words of 'Drive The Surprising Truth about What Motivates Us' ", color="red" )
plt.xticks(rotation=0, color= "blue")
plt.yticks(color='blue')
#plt.savefig("words.pdf")
plt.show()


# In[51]:


# Print the model for book 3
#tf_model[bag_of_words[2]]


# In[52]:


# Convert the tf-idf model for book 1 into a DataFrame
df_tf_idf = pd.DataFrame(tf_model[bag_of_words[2]])

# Name the columns of the DataFrame id and score
df_tf_idf.columns=["id", "tf_idf_score"]

# Add the tokens corresponding to the numerical indices for better readability
df_tf_idf['Book3_words'] = [dict[i] for i in list(df_tf_idf["id"])]

# Sort the DataFrame by descending tf-idf score and print the first 10 rows.
book3_tfidf_head = df_tf_idf.sort_values(by="tf_idf_score", ascending=False).head(5)
book3_tfidf_head


# In[53]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.bar(book3_tfidf_head["Book3_words"], book3_tfidf_head["tf_idf_score"], color = 'purple', width = 0.5)
plt.xlabel("Book3 Words", color = "green")
plt.ylabel("TF-IDF Scores", color = "green")
plt.title("Top 5 most significant words of 'Law of Attraction The Secret Power of The Universe' ", color="red" )
plt.xticks(rotation=0, color= "blue")
plt.yticks(color='blue')
#plt.savefig("words.pdf")
plt.show()


# In[54]:


# Print the model for book 4
#tf_model[bag_of_words[3]]


# In[55]:


# Convert the tf-idf model for book 1 into a DataFrame
df_tf_idf = pd.DataFrame(tf_model[bag_of_words[3]])

# Name the columns of the DataFrame id and score
df_tf_idf.columns=["id", "tf_idf_score"]

# Add the tokens corresponding to the numerical indices for better readability
df_tf_idf['Book4_words'] = [dict[i] for i in list(df_tf_idf["id"])]

# Sort the DataFrame by descending tf-idf score and print the first 10 rows.
book4_tfidf_head = df_tf_idf.sort_values(by="tf_idf_score", ascending=False).head(5)
book4_tfidf_head


# In[56]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.bar(book4_tfidf_head["Book4_words"], book4_tfidf_head["tf_idf_score"], color = 'purple', width = 0.5)
plt.xlabel("Book4 Words", color = "green")
plt.ylabel("TF-IDF Scores", color = "green")
plt.title("Top 5 most significant words of 'Payoff The Hidden Logic That Shapes Our Motivations' ", color="red" )
plt.xticks(rotation=0, color= "blue")
plt.yticks(color='blue')
#plt.savefig("words.pdf")
plt.show()


# In[57]:


# Print the model for book 5
#tf_model[bag_of_words[4]]


# In[58]:


# Convert the tf-idf model for book 1 into a DataFrame
df_tf_idf = pd.DataFrame(tf_model[bag_of_words[4]])

# Name the columns of the DataFrame id and score
df_tf_idf.columns=["id", "tf_idf_score"]

# Add the tokens corresponding to the numerical indices for better readability
df_tf_idf['Book5_words'] = [dict[i] for i in list(df_tf_idf["id"])]

# Sort the DataFrame by descending tf-idf score and print the first 10 rows.
book5_tfidf_head = df_tf_idf.sort_values(by="tf_idf_score", ascending=False).head(5)
book5_tfidf_head


# In[59]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.bar(book5_tfidf_head["Book5_words"], book5_tfidf_head["tf_idf_score"], color = 'purple', width = 0.5)
plt.xlabel("Book5 Words", color = "green")
plt.ylabel("TF-IDF Scores", color = "green")
plt.title("Top 5 most significant words of 'The Motivation Hacker' ", color="red" )
plt.xticks(rotation=0, color= "blue")
plt.yticks(color='blue')
#plt.savefig("words.pdf")
plt.show()


# #### Measuring the distance between texts 

# In[60]:


# Load the library allowing similarity computations
from gensim import similarities

# Compute the similarity matrix (pairwise distance between all texts)
similarity = similarities.MatrixSimilarity(tf_model[bag_of_words])

# Transform the resulting list into a DataFrame
similarity_df = pd.DataFrame(list(similarity))

# Add the titles of the books as columns and index of the DataFrame
similarity_df.columns = titles
similarity_df.index = titles

# Print the resulting matrix
similarity_df


# ### Comparing the contents of one book to other books

# ####  Books most similar to "Drive The Surprising Truth about What Motivates Us"

# In[61]:


#Comparing books with respect to "Drive The Surprising Truth about What Motivates Us"

# This is needed to display plots in a notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Select the column corresponding to "Drive The Surprising Truth about What Motivates Us" and 
compare = similarity_df["Drive The Surprising Truth about What Motivates Us"]

# Sort by ascending scores
compare_sorted = compare.sort_values(ascending=True)

#storing all books except "Addiction, Procrastination, and Laziness" to a variable
excluded_book = compare_sorted.iloc[[0,1,2,3]]


# Plot this data has a horizontal bar plot
excluded_book.plot.barh(x='lab', y='val', rot=0, color = "green").plot()

# Modify the axes labels and plot title for better readability
plt.xlabel("Cosine distance", color = "red")
plt.ylabel("")
plt.title("Most similar books to 'Drive The Surprising Truth about What Motivates Us'", color = "Blue")
plt.yticks(color='red')
plt.show()


# ### Performing Hierarchical clustering  based on Single, complete and average linkages.

# In[77]:


#Single linkage
cluster = hierarchy.linkage(similarity_df, 'single')

# Display this result as a dendrogram
p = hierarchy.dendrogram(Z,  leaf_font_size=12, labels=similarity_df.index,  orientation="top", leaf_rotation=90)


# In[78]:


# Complete linkage
cluster = hierarchy.linkage(similarity_df, 'complete')

# Display this result as a dendrogram
p = hierarchy.dendrogram(Z,  leaf_font_size=12, labels=similarity_df.index,  orientation="top", leaf_rotation=90)


# In[79]:


# Average linkage
Z = hierarchy.linkage(similarity_df, 'average')

# Display this result as a dendrogram
a = hierarchy.dendrogram(Z,  leaf_font_size=12, labels=similarity_df.index,  orientation="top", leaf_rotation=90)


# 
