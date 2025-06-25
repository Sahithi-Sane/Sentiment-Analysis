# Import necessary libraries
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.util import mark_negation
from nltk.corpus import sentiwordnet as swn
import inflect
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import torch
import tensorflow as tf
import gensim
from gensim.models import Word2Vec
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


string.punctuation

# Set up NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('sentiwordnet')


#initialize variables
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
p = inflect.engine()
tfidf_vectorizer = TfidfVectorizer()
count_vectorizer = CountVectorizer()




# Function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

# Function to remove original text from Google Translated data
def remove_original(text):
    translated_text = re.search(r'\(Translated by Google\)(.*?)\(Original\)', text, re.DOTALL)
    if translated_text:
        translated_text = translated_text.group(1).strip()
    return translated_text if translated_text else text

# Function to remove emojis from text
def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  
                               u"\U0001F300-\U0001F5FF"  
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F700-\U0001F77F"  
                               u"\U0001F780-\U0001F7FF"  
                               u"\U0001F800-\U0001F8FF"  
                               u"\U0001F900-\U0001F9FF"  
                               u"\U0001FA00-\U0001FA6F"  
                               u"\U0001FA70-\U0001FAFF"  
                               u"\U0001FB00-\U0001FBFF"  
                               u"\U0001F004-\U0001F0CF"  
                               u"\U0001F10D-\U0001F251"  
                               u"\U0001F004-\U0001F251" 
                               "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)

# Function to remove URLs, HTML tags, and special characters
def remove_url_html_spc(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    text = text.strip()
    return text

# Function to remove stopwords
def remove_stopwords(text):
    filtered_tokens = [token for token in text if token not in stop_words]
    return filtered_tokens

# Function for stemming
def handle_stemming(text):
    stemmed_tokens = [stemmer.stem(token) for token in text]
    return stemmed_tokens

# Function for lemmatization
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text

# Function to handle negation
def handle_negate_text(text):
    negated_tokens = mark_negation(text)
    return negated_tokens

# Function to replace numbers with words
def replace_numbers_with_words(text):
    words = text.split()
    for i, word in enumerate(words):
        if word.isnumeric():
            words[i] = p.number_to_words(word)
    return ' '.join(words)

# Function to calculate sentiment
def calculate_sentiment(tokens):
    sentiment_score = 0
    sentiment = ""
    for token in tokens:
        senti_synsets = list(swn.senti_synsets(token))
        if senti_synsets:
            # Use the average sentiment score for each sense of the word
            token_sentiment = sum([s.pos_score() - s.neg_score() for s in senti_synsets]) / len(senti_synsets)
            sentiment_score += token_sentiment
    if sentiment_score > 0:
        sentiment = "Positive"
    elif sentiment_score == 0:
        sentiment = "Neutral"
    else:
        sentiment = "Negative"
    return sentiment


# Class for data-related utilities
class dataUtils():

    def __init__(self):
        pass

    def GetGoogleReviewData(self):
        path = 'Dataset/review-Michigan_10.json'
        google_review_data = pd.read_json(path, lines=True)
        return google_review_data

# Class for data preprocessing
class preProcessing():

    def __init__(self):
        pass

    def removeEmptyReview(self,data):
        data = data.dropna(subset=['text'])
        return data

    def HandleTranslatedData(self,data):
        data["processed_review"]  = data['text'].apply(lambda x:remove_original(x))
    
    def handlePunctuation(self,data):
        data['processed_review'] = data['processed_review'].apply(lambda x: remove_punctuation(x))  
    
    def DoLowerCase(self,data):
        data['processed_review'] = data['processed_review'].apply(lambda x: x.lower())  
    
    def removeEmoji(self,data):
        data['processed_review'] = data['processed_review'].apply(lambda x: remove_emojis(x))

    def handleURLAndHTMLAndSpecialCharacter(self,data):
        data['processed_review'] = data['processed_review'].apply(lambda x:remove_url_html_spc(x))

    def handleStopWords(self,data):
        data['processed_review'] = data['processed_review'].apply(lambda x: remove_stopwords(x))
 
    #just wrote this, but I prefer doing Lemmatization over stemming
    def handleStemming(self,data):
        data['processed_review'] = data['processed_review'].apply(lambda x: handle_stemming(x))
    
    def handleLemmatizer(self,data):
        data['processed_review'] = data['processed_review'].apply(lambda x: lemmatizer(x))

    def handleNegation(self,data):
        data['processed_review'] = data['processed_review'].apply(lambda x: handle_negate_text(x))

    def handleNumericValue(self,data):
        data['processed_review'] = data['processed_review'].apply(lambda x: replace_numbers_with_words(x))

    def tokenizeText(self,data):
        data['processed_review'] = data['processed_review'].apply(lambda x: word_tokenize(x))

    #rule-based sentiment analysis
    def addSentiwordAnalysis(self,data):
        data['sentiment'] = data['processed_review'].apply(lambda x: calculate_sentiment(x))

    def joinTextData(self,data):
        data['processed_review'] = data['processed_review'].apply(lambda x: ' '.join(x))
    
# Class for text vectorization
class textVectorizer():

    def __init__(self):
        pass
    
    #for ML models we can use TFIDF , bagOfWords, spacy word embeddings for vectorization
    def handleTFIDFVectorizer(self,data):
        tfidf_matrix = tfidf_vectorizer.fit_transform(data['processed_review'])
        return tfidf_matrix

    def bagOfWords(self,data):
       count_matrix = count_vectorizer.fit_transform(data['processed_review'])
       return count_matrix

    def spacyWordEmbeddings(self,data):
        nlp = spacy.load("en_core_web_sm")
        Spacy_Embeddings = data['processed_review'].apply(lambda x: nlp(x).vector)
        return Spacy_Embeddings

    #for DL model we can use PyTorch, Keras -> word2vec
    def pytorchWord2VecEmbedding(self,data):
        sentences = data['processed_review'].tolist()
        word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)
        word2vec_matrix = torch.Tensor([word2vec_model.wv[word] for sentence in sentences for word in sentence])
        return word2vec_matrix

    def kerasWord2VecEmbedding(self,data):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(data['processed_review'])
        sequences = tokenizer.texts_to_sequences(data['processed_review'])
        padded_sequences = pad_sequences(sequences)
        return padded_sequences






    








        
        
    


 

