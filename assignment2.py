import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Define any new functions, variables, classes here
# For example, classes to create the Tree, Nodes etc
class BigramWordPredictor:
    def __init__(self):
        self.model = DecisionTreeClassifier()
        self.vocab = {}

    def preprocess_word(self, word):
        bigrams = [word[i:i+2] for i in range(len(word)-1)]
        return bigrams

    def fit_vectorizer(self, bigram_words):
        bigram_set = set()
        for bigrams in bigram_words:
            bigram_set.update(bigrams)
        self.vocab = {bigram: i for i, bigram in enumerate(bigram_set)}

    def transform(self, bigram_words):
        X = np.zeros((len(bigram_words), len(self.vocab)))
        for i, bigrams in enumerate(bigram_words):
            for bigram in bigrams:
                if bigram in self.vocab:
                    X[i, self.vocab[bigram]] += 1
        return X

    def my_fit(self, dictionary):
        bigram_words = [self.preprocess_word(word) for word in dictionary]
        self.fit_vectorizer(bigram_words)
        X = self.transform(bigram_words)
        y = np.array(dictionary)
        self.model.fit(X, y)

    def my_predict(self, bigrams):
        X = self.transform([bigrams])
        predictions = self.model.predict_proba(X)
        top_indices = np.argsort(predictions[0])[-5:][::-1]
        top_words = [self.model.classes_[i] for i in top_indices]
        return top_words

################################
# Non Editable Region Starting #
################################
def my_fit(words):
################################
#  Non Editable Region Ending  #
################################

    # Do not perform any file IO in your code
    # Use this method to train your model using the word list provided
    bigram_predictor = BigramWordPredictor()
    bigram_predictor.my_fit(words)
    return bigram_predictor  # Return the trained model

################################
# Non Editable Region Starting #
################################
def my_predict(model, bigram_list):
################################
#  Non Editable Region Ending  #
################################

    # Do not perform any file IO in your code
    # Use this method to predict on a test bigram_list
    # Ensure that you return a list even if making a single guess
    guess_list = model.my_predict(bigram_list)
    return guess_list