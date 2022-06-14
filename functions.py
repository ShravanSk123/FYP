import nltk
import re
import heapq
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation

from bs4 import BeautifulSoup
from urllib.request import urlopen

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Text categories
categories = ['Economy & Business', 'Diverse News', 'Politics', 'Sports', 'Technology']


# Text summarizer
def nltk_summarizer(input_text, number_of_sentences):
    number_of_sentences = int(number_of_sentences)
    stopWords = set(nltk.corpus.stopwords.words("english"))
    word_frequencies = {}
    k = nltk.word_tokenize(text_prepare(input_text))
    for word in k:  
        if word not in stopWords:
            if word not in punctuation:
                try:
                    word_frequencies[word] += 1
                except:
                    word_frequencies[word] = 1

    maximum_frequency = max(word_frequencies.values())

    # finding weighted frequency
    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequency)

    #print(word_frequencies)

    max_sent_len = 35

    sentence_list = nltk.sent_tokenize(input_text)
    sentence_scores = {}  
    for sent in sentence_list:  
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < max_sent_len:
                    try:
                        sentence_scores[sent] += word_frequencies[word]
                    except:
                        sentence_scores[sent] = word_frequencies[word]

    summary_sentences = heapq.nlargest(number_of_sentences,sentence_scores, key = sentence_scores.get)
    #print(summary_sentences)
    summary = ' '.join(summary_sentences)
    return summary

# Loading the dataset
data = pd.read_csv(r"dataset/bbc_news_dataset.csv")
data = data.replace("entertainment", "diverse news")
data = data.replace("business", "economy & business")

#Preprocessing steps

# 1. Remove all URL links like http and www
def delete_links(input_text):
        #out_text = re.sub(r'http\S+', '', input_text)
        out_text = re.sub("((http|https)://)*(www.)?" +
            "[a-zA-Z0-9@:%._\\+~#?&//=]" +
            "{2,256}\\.[a-z]" +
            "{2,6}\\b([-a-zA-Z0-9@:%" +
            "._\\+~#?&//=]*)", "", input_text)
        return out_text


# 2. Remove HTML tags present in the text
from bs4 import BeautifulSoup
def strip_html_tags(input_text):
    soup = BeautifulSoup(input_text, "html.parser")
    out_text = soup.get_text(separator=" ")
    return out_text


# 3. Convert some foreign characters in text like to normal characters
# Ex: à, è, ù -> a, e, u
import unidecode
def convert_foreignchars(input_text):
    out_text = unidecode.unidecode(input_text)
    return out_text


# 4. Removing extra symbols
def remove_irrelevant_chars(input_text):
    out_text = re.sub('[^a-zA-Z]',' ',input_text)
    return out_text


# 5. Removing extra whitespaces in text
def remove_whitespace(input_text):
    out_text = input_text.strip()
    return " ".join(out_text.split())


# 6. Transform cases on the text (convert to lower-case)
def transform_cases(input_text):
    out_text = input_text.lower()
    
    return out_text


# 7. Delete repeated characters
# Ex: coool -> cool 
def delete_repeated_characters(input_text):
    pattern  = r'(.)\1{2,}'
    out_text = re.sub(pattern, r"\1\1", input_text)

    return out_text


# 8. Lemmatization
# Ex: saying -> say, took -> take
from nltk.stem import WordNetLemmatizer
def Lemmatization(input_text):
    k = WordNetLemmatizer()
    word_tokens = word_tokenize(input_text)

    out_text = []
    for w in word_tokens:
        out_text.append(k.lemmatize(word = w, pos = 'v'))
    
    return ' '.join(out_text)


# 9. Remove stopwords
# Ex: the, an, in, for
def remove_stopwords(input_text): 
    stop_words = set(stopwords.words('english'))
 
    word_tokens = word_tokenize(input_text)
 
    out_text = []
 
    for w in word_tokens:
        if w not in stop_words:
            out_text.append(w)
    #print(word_tokens)
    
    return ' '.join(out_text)


def text_prepare(input_text):
    out_text = delete_links(input_text)
    out_text = strip_html_tags(out_text)
    out_text = convert_foreignchars(out_text)
    out_text = remove_irrelevant_chars(out_text)
    out_text = remove_whitespace(out_text)
    out_text = transform_cases(out_text)
    out_text = delete_repeated_characters(out_text)
    out_text = Lemmatization(out_text)
    out_text = remove_stopwords(out_text)
    return out_text

def text_prepare2(input_text):
    out_text = remove_whitespace(input_text)
    return out_text

# Apply text prepare function for dataset
data['Processed Text'] = data['Text'].apply(text_prepare)

# Label encoding
label_encoder = LabelEncoder()
data['Category Encoded'] = label_encoder.fit_transform(data['Category'])

# Splitting the data into train and test
# 80% for training, 20% for testing & validation
X_train, X_test, y_train, y_test = train_test_split(data['Processed Text'], data['Category Encoded'], test_size=0.2, random_state=0)

# TF-IDF vectorizer:
def tfidf_features(X_train, X_test):
    tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2))
    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_test = tfidf_vectorizer.transform(X_test)
    return X_train, X_test

features_train, features_test = tfidf_features(X_train, X_test)

# Summarize and classify for input text:
def TSC(input_text, number_of_sentences, model_name):
    # new_text = text_prepare(input_text)
    summary_text = nltk_summarizer(input_text, number_of_sentences)
    input_text_arr = [text_prepare(input_text)]
    features_train, features_test = tfidf_features(X_train, input_text_arr)
    text_prediction = model_name.predict(features_test.toarray())
    text_category = categories[text_prediction[0]]
    return summary_text, text_category

# To fetch data from URL
def fetch_data(url):
    page = urlopen(url)
    soup = BeautifulSoup(page)
    fetched_text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    return fetched_text