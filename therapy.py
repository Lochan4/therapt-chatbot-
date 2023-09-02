import random
import re
import ssl

import nltk
import pandas as pd

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download('omw-1.4')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
emotion_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear"}

def therapy_chatbot():

opposite_map = {0: 1, 1: 0, 2: 3, 3: 2, 4: 1}
negation_list = ["not"]
data = pd.read_csv("response_sheet (1).csv").values
result = {}
for key, value in data:
    if key not in result:
        result[key] = [value]
    else:
        result[key].append(value)

data = pd.read_csv(r"training.csv")
description_list = []

# data processing
for description in data.text:
    # filter words
    description = re.sub("[^a-zA-Z]", " ", description)
    # change it to lower case
    description = description.lower()

    # tokenizing word
    description = nltk.word_tokenize(description)
    # # removing stop words
    # description = [word for word in description if not word in set(stopwords.words("english"))]

    # lemmentize - faster to fast
    lemmatizer = WordNetLemmatizer()
    description = (lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word, "n"), pos="v"), pos="a") for
                   word in description)

    description = " ".join(description)
    description_list.append(description)

x = description_list
y = data.label.values

# Create a feature extractor
vectorizer = CountVectorizer()

# Convert text data into numerical feature vectors
X = vectorizer.fit_transform(x)

# Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X, y)

print("Welcome to the therapy chatbot!")
print("How are you feeling today?")
while True:
    user_input = input("")
    if user_input.lower() == "quit":
        print("Thank you for chatting. Goodbye!")
        break
    user_input = re.sub("[^a-zA-Z]", " ", user_input)
    user_input = user_input.lower()

    user_input = nltk.word_tokenize(user_input)

    user_input = (lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word, "n"), pos="v"), pos="a") for
                   word in user_input)

    user_input = " ".join(user_input)
    user_input_vector = vectorizer.transform([user_input])
    y_pred = classifier.predict(user_input_vector)[0]
    split_user_input = user_input.split(" ")
    # checking for negation
    for word in split_user_input:
        if word in negation_list:
            y_pred = opposite_map[y_pred]
            break
    response_list = result[y_pred]
    print(random.choice(response_list))