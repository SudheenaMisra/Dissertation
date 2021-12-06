# Dissertation
My MSc Data Science Dissertation Project. 
For the PDF version of the full report check this repository 
## Content Based Motivational book recommendation using Natural Language Processing and Clustering
##### Data collection
The first step I followed in conducting my analysis was data collection. I wanted 
to do text mining for some motivational books. When I searched the books online, 
I found most of the sites provide paid downloads only. So I searched about the 
free sites on YouTube to download books. I found the following site https://z-lib.org/
which provides free access to books. When we go through this link, we can find 
three other links, "Books," "Articles," and "sign in/up," on the page. For analyzing 
the text first, it is better to convert the books into .txt format. The books on this 
site, by default, are in .epub format; most of the system doesn't support this 
format. But this site has an option to convert to a .txt file and download if we sign 
up. So I signed up to this site first, went to the "Books" section, and searched the 
keyword "Motivation." The result shows lots of motivational books. I randomly 
selected five books from 5 different authors from the result page and downloaded 
them into .txt format for my analysis. The books I selected are 
i) “Addiction, Procrastination, and Laziness A Proactive Guide to the 
Psychology of Motivation” by Roman Gelperin
ii) “Drive The Surprising Truth About What Motivates Us" by Daniel H. Pink 
iii) “The Law of Attraction, The Secret Power Of The Universe” by Jenny 
Loveless
iv) “Payoff The Hidden Logic That Shapes Our Motivations” by Ariely Dan
v) “The Motivation Hacker by Nick Winter

#### Text Mining
I started my text mining by converting all the texts into lowercase. The remaining 
steps I followed are listed below.
##### Tokenization
Tokenization is the process of splitting the texts into individual tokens. These 
tokens can be a sentence, word, or character. That means each line of the text 
either contains one sentence if it is a sentence token, or each line contains only 
one word if it is a word token or each line contains only one character if it is a 
9
character token. If we want to do sentence tokens, we can split the texts with 
respect to a full stop("."). So that each line contains one sentence, or if we want 
to do the words token, we can split with respect to "space" So that each line 
contains only one word. Likewise, we can split the text so that each line contains 
only one letter, which is called character tokens. In most natural language 
applications, the word tokens are considered most valuable because we could 
explore and extract each word's sentiment individually. But every token is 
beneficial depending on the tasks we want to perform with the texts. For my 
analysis, I am converting all these texts into words tokens. Tokenization also 
removes the punctuations in the texts.
#### Defining and Removing Stop Words
If we go through any piece of text, these texts can be from any books, 
articles, opinions, conversations, and whatnot, we could find some of the words 
dominate in every text. The words such as "the," "a," "as," "which," "has," etc., 
are the most common words we could find in every text. The collection of these 
words are called stop words. Since these words don't add much significance to 
the context of our text analysis, it is recommended to remove all these words 
before the analysis, which will reduce the size of the texts and also facilitate less 
running time, thereby enhancing the quality of the code. By removing the stop 
words, we are helping our analysis focus on the significant pieces of information 
rather than the low-level pieces of information. So our analysis will be in point to 
point rather than including all the useless pieces of information, thereby 
facilitating the model training easier.[3]
To remove these stop words, I have searched for the lists of stop words 
in google. I found a couple of links. Some of them contain more number stop 
words, and others contain less number of stop words; I choose the one with more 
number stop words. I copied all these stop words and created a list of stop words 
in python. I named this variable stop_words.


##### Stemming
Stemming is transforming a word to its base or root form called a "stem" by 
reducing the inflections in those words. Wikipedia states the definition of 
inflection as, “In linguistic morphology, inflection (or inflexion) is a process 
of word formation, in which a word is modified to express different grammatical 
categories such as tense , case, voice person, number, gender, mood, animacy, 
and definiteness. An inflection expresses grammatical categories with affixation 
(such as prefix, suffix, infix, circumfix, and transfix), 
11
apophony ((as Indo-European ablaut), or other modifications”[4].
Stemming is the most common practice of text mining in natural language 
processing, search engines, and information retrieval applications. This 
algorithm unifies the different forms of a single word to its root form. The 
authors sometimes use different words to explain the same concept. So 
identifying these words and converting them to their base form plays a vital role 
in text mining.
There are three main types of stemming algorithms. They are truncating methods 
(affix removal), statistical methods, and mixed methods. All these three 
algorithms have several sub-categories as well. Since this is a vast topic to 
explore, I will only address that one stemming algorithm that I will be using in 
this project, the Porters Stemmer algorithm.[5] 
Porters Stemmer algorithm is a popular stemming algorithm with less error 
rate and produces the best output compared to other stemmers. This belongs to 
the sub-category of truncating methods or affix removal. Let's understand the 
function of this algorithm with the help of a diagram.

![image](https://user-images.githubusercontent.com/76818040/144921572-8cd0fed4-48cd-4f80-b435-9ba4a831540c.png)
##### Stemming results
I am trying to visualize how it looks like after the stemming. I have assigned the output of the stemming into a variable called "stems." This "stems" has five indexes, and each index consists of the list of stem words of each book, in Fig 13. I only marked the first ten stems of each book. I draw this figure manually to communicate better.
![image](https://user-images.githubusercontent.com/76818040/144921902-37f6edee-e9ae-4878-8e0a-e64b653862f4.png)

##### 	Generating a Term Frequency - Inverse Document Frequency (TF- IDF) model
In the previous step, we have counted the most common words in each book. Even though some words occur more times in a book or document, it doesn't mean that is the word that is more significant in that book. If we shouldn't have removed the stop words, they might have been repeated in every book and may provide us with the wrong results. Even though we removed the stop words, we couldn't say that we obtained the most important words. Term Frequency – Inverse Document Frequency(TF-IDF) algorithm helps us identify the most important words of a particular document by comparing that specific words in all other documents. This algorithm is advantageous in search engines, classifying texts, and finding keywords. This significant word may be the keyword for that particular book. Suppose if we searched a word in google, for example, how the search algorithm work is that they will return the results of the documents which have the high TF-IDF scores for that particular word. So what I am going to do is calculate the TF-IDF scores of each word in each book. Before that, I will explain this algorithm in more detail.
Mathematically, TF-IDF scores are calculated as follows [13];
TF_IDF = TF * IDF
Where TF is the Term Frequency and IDF is the inverse document frequency. Let’s consider three documents d1, d2, and d3, in a corpus (as we know, the corpus is a collection of documents, and the document is a collection of words) for an example.
d1  =  “Life will change when you change."
d2  =  “If you can dream it you can achieve it."
d3  = “Life, life , life  everyone have only one life.”

Let's consider the word "life" in the three documents and calculate its term frequency, inverse document frequency, and TF-IDF scores.
Term Frequency(TF)
Term frequency is calculated per document. ‘Term frequency is defined as the number of times a word or term appears in a document[10]. The word “life” has appeared once in d1, zero times in d2, and four times in d3. So the term frequency is as follows [11].

		
TF(“life) of d1 = 1
TF(“life”) of d2 = 0
TF(“life”) of d3 = 4

Inverse Document Frequency (IDF)
IDF is calculated as follows;
IDF =  log⁡((Total number of documents )/(number of documents in which the  word is present))
Here we have three documents in our corpus and among which 2 of the documents contain the word "life."
IDF(“life”) of d1 = log (3/2) = 0.176
IDF(“life”) of d2 = log (3/2) = 0.176
IDF(“life”) of d3 = log (3/2) = 0.176
From the above example, we can see that the IDF is constant per corpus.

Calculation of TF_IDF (“life”) [12]
  TF-IDF = TF * IDF
TF_IDF (“life”) of d1 = 1 * 0.176 = 0.176
TF_IDF (“life”) of d2 = 0 * 0.176 = 0
TF_IDF (“life”) of d3 = 4 * 0.176 =0.704

The above calculation shows that document 3 has the highest TF-IDF scores, which suggests that the word "life" is more significant to d3 than d1.
Measuring the similarity between the texts in books
Now we have our TF-IDF model results which give the TF-IDF scores of the most specific stemmed words in each book. The next step is to measure the distance between the texts or, in other words finding out how similar each book is. In order to perform this, we need to select the best choice of similarity. There are lots of options available when considering measuring the similarity, but all those may not lead to accurate results since all similarity measures available serve a different purpose and are only applicable in specific scenarios. Let's consider the three most popular similarity distance measures in machine learning, and they are Manhattan distance, Euclidean distance, and Cosine Similarity. Before deciding how to choose the similarity measure, it’s important to know how we can represent the documents(books) and words mathematically to calculate it.

In NLP, the documents are represented in multi-dimensional space as vectors where the words are their coordinates, and they will decide where to place these documents in space depending on some features of these words in each document ( In most scenarios, these features may be the number of occurrences of these words and the TF-IDF scores). Suppose if we have two documents that consist of only two words, it can be visualized in a two-dimensional space or if we have more than two documents, say 100 documents, but if each document contains only two words, still it can be represented in 2-dimensional space. But if the words are 3 in each document, they can be represented in 3-dimensional space. Here the dimension of the space where the documents are placed will be determined by the unique words in each document, not the number of documents we have. We can visualize up to 3-dimensional space, i.e., up to 3 words. But we all know that might not be the ideal case. Every document that can be any piece of text like books, articles, essays, etc., contains more words, most probably more than 100 words. We couldn't even imagine a 100 dimensional space and visualize it on paper. But still, we can employ the same methods and formulas we used in 2-D space or 3-D space to measure the distance and orientation of these documents in any multi-dimensional space. Let's see an example of how two documents that contain only two words can be visualized in a two-dimensional space.
        
        For example, consider two documents, D1 and D2, where these two documents 
contains only two words, and they have the same words but the number of
occurrences of each word are different in 2 documents.

D1 = { “Life” : 5, “Motivation”: 3}
D2 = {“Life” : 6, “Motivation”: 8 }

Here the word "life" has occurred five times in D1 and six times in D2, whereas the word "motivation" has occurred three times in D1 and eight times in D2. In the fig below, we can see that I represented these two documents in a 2-dimensional plane with the number of occurrences of the words "life" in the Y-axis and the number of occurrences of "Motivation" in the X-axis. So we can represents the documents D1 and D2 as vectors  mathematically in terms of (x,y) coordinates as D1(x,y) = (3,5) and D2(x,y) = (8,6) 
     ![image](https://user-images.githubusercontent.com/76818040/144925081-80605fb0-32df-4582-a40c-6c7c5e021a26.png)        

Now we understand how the documents are represented in space. The next question will be how we calculate the similarity between the documents. As at the beginning of this section, I have mentioned about three popular distance measures. Let's understand these distance measures with the help of a diagram.

![image](https://user-images.githubusercontent.com/76818040/144922463-e13f7747-cf18-4d58-8188-e6f4334f754a.png)

From the fig., we need to calculate the distance between the document vectors D1 and D2, and we can calculate these distances in three ways; the First one is the Manhattan distance, the second Euclidean distance, and the third Cosine distance.
Manhattan distance: This distance is also called city block distance as it is calculated by means of horizontal and vertical distances from each document vector.
 As per the diagram, Manhattan distance = a + b
Where ‘a’ is the vertical distance, and ‘be is the horizontal distance. 

Euclidean distance: As we can see from the figure, this is the shortest distance between the two document vectors. Pythagoras theorem is used to calculate this distance.
Both the Manhattan distance and Euclidean distance are the special cases of Minkowski Distance. I am not going to explain more about these distances because it is not relevant to this project. 

Both these distances are useful in scenarios where the magnitude of the vectors makes more sense than their orientations. In other words, when dealing with numerical quantities, these distances are more useful. But if we use these distances in our project to measure the similarities of books, these algorithms may classify the books like all the shorter books are similar or all the lengthy books are similar, because we know that when analyzing texts, some books are shorter, some others are longer. If a book is lengthy, some words may repeat more times than they repeat in short books. So, we don't need to arrive at the wrong conclusion with this. We need to compare the contents regardless of their length in general. So, the Cosine distance will be the better choice. It is calculated by means of the angle between the two document vectors. It focuses on the orientations of the vectors in space regardless of their magnitude.

##### Calculation of Cosine distance
Cosine distance can be explained mathematically as. 
Cosine similarity, cos⁡θ
cos⁡θ=(D ⃗_1⋅D ⃗_2)/|(D_1 ) ⃗ ||D ⃗_2 | 
Cosine distance = 1- cosine similarity = 1- cos(θ)	
Let's consider an example with more words. This cannot visualize because it is in multi-dimensional space.
D1 = “ I like chocolate, but I like spices more.”
D2 = "She also like chocolate."
 So we have two documents D1 and D2, with more words. Here we have eight unique words in total, and some words occur in both documents. So these documents can be represented as vectors in 8-dimensional space. Before measuring the cosine distances, we will do preprocessing so any punctuations they have all get removed, and also the texts will be in lowercase as well. This is just an example of how it works.

Documents	“I”	“like”	“chocolate”	“but”	“spices”	“more”	“she”	“also”
D1	2	2	1	1	1	1	0	0
D1	0	1	1	0	0	0	1	1

So D1 and D2 can be represented as vectors, as shown below.
D1 = [2,2,1,1,1,1,0,0]
D2 = [0,1,1,0,0,0,1,1]
and the cosine similarity can be calculated from above formula as 
cos⁡θ  =   (( 2*0 + 2*1 + 1*1 + 1*0 + 1*0 +1*0 + 0*1 +0*1 ))/(sqrt(2^2+2^2+1^2+1^2+1^2+1^2+0^2+0^2 )* sqrt(0^2+1^2+1^2+0^2+0^2+0^2+1^2+1^2 ) ) = 0.375
i.e., cosine similarity (D1, D2) = 0.375
cosine distance = 1- 0.375 = 0.625
The pair-wise distance can be calculated, and it can be represented in a matrix called the similarity matrix by applying the cosine similarity formula. The similarity matrix is represented in a square form where the rows and columns will be the same. Refer to fig; below.
![image](https://user-images.githubusercontent.com/76818040/144922949-e81802db-741a-4414-9cda-60280489e0ea.png)
From the figure, we can see the diagonal elements of the matrix are 1. This is because the similarity is calculated between the documents themselves, that is, between D1 and D1, the D2 and D2. Since they are the same and the angle between D1 and D1 is zero, likewise for D2. Hence cos 0 = 1. If cos 0 is 1, which implies that they are the same in orientation but may be or may not be different magnitude, but we can say that they are similar. The cosine function is equal to 0 when the angle between them is 90 degrees. This indicates that both the document vectors are dissimilar. Cos (D1, D2) represents the cosine distance between the documents D1 and D2; this will be calculated using the formula I have mentioned. We see that the elements of the matrix above and below the diagonal elements are the same; this is how a similarity matrix is represented. When we calculate the cos (D1, D2), if we get a value close to 1, it means that these documents are similar, and if we get a value close to zero, it means they have more dissimilar contents.  Here in this example, we only have two documents; hence the dimension of this matrix is 2x2, and it has four elements (i.e., 2^2).  When the number of documents changes, the dimension of this matrix changes accordingly.  In this project, our five books are the documents, so our similarity matrix will have a dimension of 5x5 (i.e., 5^2= 25 elements in total).
	In the above example, I represent the vectors in space based on the word count for simplicity. But in this project and most of the cases, it is represented, and the cosine distance is calculated based on the word's TF-IDF scores instead of its word count because that makes more sense. 

##### Cosine Similarity matrix of the 5 Books
In the methodology section, I have explained how we can calculate the cosine similarity matrix. Gensim library also has the inbuilt functions to calculate this matrix. The similarity matrix of our five books is calculated as shown below.
![image](https://user-images.githubusercontent.com/76818040/144923238-641264d9-5449-49a6-911c-388a1f37e4c9.png)
![image](https://user-images.githubusercontent.com/76818040/144923258-020d86bc-30cf-48f9-b462-e4a257e7ebfd.png)


##### Clustering to group similar books together
Clustering is the most commonly used algorithm in Unsupervised Machine Learning. Clustering is the process of identifying similar groups in the data points. According to Clustering, each observation or element in one group exhibits stronger similarities between the members of the same group and share dissimilarities between the members of the other group [1]. Clustering is not just a single algorithm but it is a collection of several algorithms. There are different types of Clustering. I will be using Hierarchical Clustering for this project. 
	“Hierarchical clustering (also called hierarchical cluster analysis or HCA) is a method of cluster analysis which seeks to build a hierarchy of clusters”[Wikipedia]. In order to achieve hierarchical Clustering, we need to apply certain criteria to this algorithm, and these criteria will decide how to group the data observation. These criteria are called distance similarity or dissimilarity measures. Usually, any of the similarity distance matrices such as Euclidean distance, Manhattan distance, Cosine distance, etc., will be used in this scenario.  The shape and outcome of the hierarchical Clustering depend on the choice of the matrix. In the previous section, we already selected the appropriate similarity distance that is our cosine similarity distance, and we already have our cosine similarity matrix with us. So in this step, we just need to pass this matrix to the clustering algorithm. 
	The results of hierarchical Clustering are usually represented in the form of the dendrogram. A Dendrogram is a pictorial representation of data points or the documents arranged in a bottom-up approach which shows the hierarchical relationship between the data points in that cluster[19].
	
##### Hierarchical Clustering output 

 We see that the elements above and below the diagonal elements are the same for a similarity matrix. For simplicity, we can represent our similarity matrix as shown below. The figure 24 is the same cosine similarity matrix we obtained from the previous section. I just replaced the name of the books with D1, D2, D3, D4, and D5 for simplicity.
Where, D1: 'Addiction, Procrastination, and Laziness'
		D2: 'Drive, the Surprising Truth about What Motivates Us'
	D3: 'The Law of Attraction, The Secret Power of The Universe'
	D4: 'Payoff, The Hidden Logic That Shapes Our Motivations'
	D5: 'The Motivation Hacker'
 Since we are using the cosine similarity distance, the distances which are closer to 1 are the minimum distance.
Iteration 1: First iteration is same for all the 3 linkage algorithms
Identifying the minimum pair-wise similarity distance(highlighted in yellow color)

Fig 22. Clustering algorithm iteration 1
	D1	D2	D3	D4	D5
D1	1				
D2	0.045	1			
D3	0.018	0.038	1		
D4	0.019	0.165	0.039	1	
D5	0.028	0.057	0.027	0.032	1

In the figure 22, I have highlighted a distance in yellow color, which is the value closer to 1 when compared to other values. From matrix we already identified the minimum distance as 0.165. So, we can see that this value is actually the cosine distance between D2 and D4(i.e., 'Drive, the Surprising Truth about What Motivates Us’ and 'Payoff, the Hidden Logic That Shapes Our Motivations.' So, these two books form a cluster in the first Iteration of each linkage. 
Let's see how the complete linkage algorithm works.
#### Clustering based on Complete Linkage Complete Linkage calculation steps
##### Iteration 1
D2 and D4 form a cluster in the first Iteration. So, the similarity matrix initially was a 5x5 matrix which now converges into a 4x4 matrix. We can denote the newly formed cluster as (D2, D4). (Refer fig 22)
###### Iteration 2
we need to update the similarity matrix based on the complete linkage considering the new clusters formed. Only the values affected by the new cluster (D2, D4) need to be re-calculated; other values remain the same.
 
Calculating the maximum distance between (D2, D4) and D1

Max {(D2, D4), D1} = Max {(D2, D1), (D4, D1)} 
				  = Max {0.045, 0.019} (from fig. 22)	
				  = 0.019
Remark: 0.045 is the minimum distance because it is cosine similarity (value close to one is minimum (not close to zero) for cosine similarity) So we choose 0.019 as the maximum distance between the two.

Calculating the maximum distance between (D2, D4) and D3

Max {(D2, D4), D1} = Max {(D2, D3), (D4, D3)} 
				  = Max {0.038, 0.039} (from fig. 22)	
				  = 0.038

Calculating the maximum distance between (D2, D4) and D5

Max {(D2, D4), D1} = Max {(D2, D5), (D4, D5)} 
				  = Max {0.057, 0.032} (from fig. 22)	
				  = 0.032
The similarity matrix can be updated as follow

Fig 23. Complete linkage iteration 2
	D1	(D2, D4)	D3	D5
D1	1			
(D2,D4)	0.019	1		
D3	0.018	0.038	1	
D5	0.028	0.032	0.027	1

###### Iteration3
Repeating the above steps until the algorithms converge
Here the least distance is identified as 0.038, So D3 ('Law of Attraction The Secret Power of The Universe') joins the cluster (D2, D4) in this Iteration. So the new cluster formed can be denoted as ((D2, D4), D3).
 
Calculate the maximum distance between ((D2, D4), D3) and D1.

Max {((D2, D4), D3), D1} = Max {((D2, D4), D1), (D3, D1)} 
				  = Max {0.019, 0.018} (from fig. 23)	
				  = 0.018

Calculate the maximum distance between ((D2, D4), D3)  and D5.

Max {((D2, D4), D3), D5} = Max {(((D2, D4), D5)), (D3, D5)} 
				  = Max {0.032, 0.027} (from fig. 23)
				  = 0.027


According to the re-calculated values, the similarity matrix can be updated as follows;

Fig 24. Complete linkage iteration 3
	D1	((D2, D4), D3)	D5
D1	1		
((D2,D4), D3)	0.018	1	
D5	0.028	0.027	1
##### Iteration 4

The minimum distance is identified as 0.028 and which is the distance between D1 and D5, So these two books form another cluster. It can be denoted as (D1, D5)

Calculating the maximum distance between ((D2, D4), D3) and (D1, D5)

Max {((D2, D4), D3), (D1, D5)} = Max {((D2, D4), D3), D1), ((D2, D4), D3), D5)} 
				  = Max {0.018, 0.027} (from fig. 24)	
				  = 0.018

According to the re-calculated values, the similarity matrix can be updated as follows;
Fig 25. Complete linkage iteration 4
	(D1, D5)	((D2, D4), D3)
(D1, D5)	1	
((D2,D4), D3)	0.018	1
Now the minimum distance is identified as 0.018, and at this point, the algorithm terminates, and another big cluster will be formed with all these books.  The results of this algorithm can be represented as a dendrogram as shown below.

Fig 26. Complete linkage Dendrogram
![image](https://user-images.githubusercontent.com/76818040/144923818-50fff41d-e790-441b-9ea5-a99e63b5a08c.png)

As we can see, the dendrogram follows a bottom-up approach. In the first iteration books, "Drive" and "Payoff" forms a cluster, then in the second Iteration, "law of attraction" Joins these cluster, then, in the third Iteration, another cluster of the other two remaining books are formed. Then in the last Iteration, a cluster is formed with all books.  As we may notice that the books "Payoff" and "Drive" have more similar contents and "law of attraction contains more unique ideas than all other books, but it has more similarity towards "payoff" and "Drive" than the other two books. While the other books, "Addiction" and "motivation hacker" have somewhat similar contents, so they belong to the same cluster.

#### Clustering based on Single Linkage 
Single Linkage calculation steps
Single linkage is also calculated like the complete linkage, but instead of taking the Max{distance} distance, we take the Min{distance} 

##### Iteration 1
D2 and D4 form a cluster in the first Iteration (Refer fig 22).

##### Iteration 2
we need to update the similarity matrix based on the single linkage considering the new clusters formed. Only the values affected by the new cluster (D2, D4) need to be re-calculated; other values remain the same.
 
Calculating the minimum distance between (D2, D4) and D1

Min {(D2, D4), D1} = Min {(D2, D1), (D4, D1)} 
				  = Min {0.045, 0.019} (from fig. 22)	
				  = 0.045
Remark: 0.045 is the minimum distance because it is cosine similarity. 

Calculating the maximum distance between (D2, D4) and D3

Min {(D2, D4), D1} = Min {(D2, D3), (D4, D3)} 
				  = Min {0.038, 0.039} (from fig. 22)	
				  = 0.039

Calculating the minimum distance between (D2, D4) and D5
	
Min {(D2, D4), D1} = Max {(D2, D5), (D4, D5)} 
				  = Max {0.057, 0.032} (from fig. 22)	
				  = 0.057
The similarity matrix can be updated as follow

Fig 27. Single linkage iteration 2
	D1	(D2, D4)	D3	D5
D1	1			
(D2,D4)	0.045	1		
D3	0.018	0.039	1	
D5	0.028	0.057	0.027	1

Iteration3
Repeating the above steps until the algorithms converge

Here the least distance is identified as 0.057, So D5 (‘The Motivation Hacker') joins the cluster (D2, D4) in this Iteration. So, the new cluster formed can be denoted as ((D2, D4), D5).
 
Calculate the minimum distance between ((D2, D4), D5) and D1.

Min {((D2, D4), D5), D1} = Min {((D2, D4), D1), (D5, D1)} 
				  = Min {0.045, 0.028} (from fig. 27)	
				  = 0.045s

Calculate the minimum distance between ((D2, D4), D5) and D3.

Min {((D2, D4), D3), D5} = Min {(((D2, D4), D5)), (D5, D3)} 
				  = Min {0.039, 0.027} (from fig. 27)
				  = 0.039

According to the re-calculated values, the similarity matrix can be updated as follows;
Fig 28.  Single linkage iteration 3
	D1	((D2, D4), D5)	D3
D1	1		
((D2,D4), D5)	0.045	1	
D3	0.018	0.039	1
Iteration 4

The minimum distance is identified as 0.045 and which is the distance between D1 and ((D2, D4), D5), So these books form another cluster. It can be denoted as (((D2, D4), D5), D1)

Calculating the minimum distance between (((D2, D4), D5), D1) and D3

Min {((D2, D4), D3), (D1, D5)} = Min {((D2, D4), D5), D3), (D1, D3)}
				  = Min {0.039, 0.018} (from fig. 28)	
				  = 0.039

According to the re-calculated values, the similarity matrix can be updated as follows;
Fig 29. Single linkage iteration 4
	(((D2, D4), D5), D1)	D3
(((D2, D4), D5), D1)	1	
 D3	0.039	1
Now the minimum distance is identified as 0.039, and at this point, the algorithm terminates, and another big cluster will be formed with all these books.  The results of this algorithm can be represented as a dendrogram as shown below.

	          Fig 30. Single linkage Dendrogram
![image](https://user-images.githubusercontent.com/76818040/144924360-07d51cd1-2cfd-444d-b2c2-20a4f96bd917.png)

#### Discussion and Conclusion

To conclude, in this paper, I have investigated and explored some most popular techniques in NLP and Clustering. To conduct my study, I have downloaded five books from an online site and performed some preprocessing to transform the data for analysis. I have created a BOW to only keep the unique words in each book and generated a TF-IDF model using this BOW model. According to the generated TF-IDF scores, cosine similarities between the books are calculated. Using this cosine similarity, I have performed hierarchical Clustering. Then I generated the dendrograms with three different linkage algorithms in hierarchical Clustering. Single linkage and average linkage have the same dendrogram as output, but for complete linkage it was quite different since it explains in a wide perspective. 

From the output from the Clustering, we can conclude that the books "Payoff" and "Drive" are the closest relatives also the books "Addiction" and "Motivation hacker" also have a little similarity. The book "Law of attraction" has is more dissimilar to all other books, but it has a little lean towards "Drive" or "Payoff" compared to other books. In general, we can conclude that even though all the books I have selected belong to the same category of motivation books. They all have unique content than similar content. Some books have expressed some similar ideas, but most of the contents are dissimilar. All these books are from different authors. So, all of them express the view towards how to motivate people from a different perspective. Some similarities have occurred because all of their targets are the same, that is, to give motivation.  In future works, I would like to consider analyzing the books from the same author and also like to do some research about how the google search queries or any other search engines recommend the result page. 




