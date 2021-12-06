# Dissertation
My MSc Data Science Dissertation Project
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

After I downloaded these books, I went through all these text files quickly without 
reading just a quick scroll down; I found Some of the texts, in the beginning, 
stating the copyright permissions of each book and some acknowledgments and 
some recommendations at the end of each book. I think these texts are 
insignificant for my analysis, so I manually removed all those texts before I 
uploaded these .txt files into miniconda software for my analysis.

Text Mining
I started my text mining by converting all the texts into lowercase. The remaining 
steps I followed are listed below.
4.3.1Tokenization
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
4.3.2 Defining and Removing Stop Words
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


4.3.3 Stemming
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
