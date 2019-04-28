import numpy as np
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok
#from mosestokenizer import MosesDetokenizer
from nltk.corpus import brown
from nltk import ngrams
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split


class Segmenter:

    def __init__(self):
        '''Loads data for training and testing a CRF model'''
        self.crf = self.load_model('crf.model')
        self.tokens_lm = (self.load_model('postags_language.model'), 3)
        self.postags_lm = (self.load_model('tokens_language.model'), 5)

    def gen_labels(self, sent, pos):
        '''Returns a lists of labels: creates a list of labels 'S' and inserts 'P' into given position '''
        ''' 'S' if the token if followed by space and 'P' if the token if followed by a period'''
        labeled_sents = ['S'] * len(sent)
        labeled_sents[pos] = 'P'
        return labeled_sents

    def load_data(self):
        '''Instantiates train and test data from joined, labeled, and POS tagged sentences from Brown corpus'''
        # Read data corpus into sents
        sents = brown.sents()

        # Make sure there's an even number of sents
        length = len(sents)
        length = length if (length % 2 == 0) else (length - 1)

        data = []
        for i in range(0, length, 2):
            if (len(sents[i]) < 2 or len(sents[i + 1]) < 2):
                continue
            # Join every two sentence
            joined_sent = sents[i][:-1] + sents[i + 1][:-1]

            # Get labels
            labels = self.gen_labels(joined_sent, len(sents[i]) - 2)

            # Perform POS tagging
            tagged = nltk.pos_tag(joined_sent)

            # Take the token, POS tag, and its label
            data.append([(token, pos, label) for label, (token, pos) in zip(labels, tagged)])

        X = [self.sent2features(sent) for sent in data]
        y = [self.sent2labels(sent) for sent in data]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)
        print('Data loaded.')

    def is_perplexity_decreasing(self, items, model, n):
        '''Returns true if per-word perplexity of the sentence decreases if a period is inserted in front of the items'''
        arr = []
        for i in (items, items + ['.']):
            padded_grams = ngrams(i, n, pad_left=True, left_pad_symbol='<s>')
            arr.append(model.perplexity(padded_grams))
        return arr[0] > arr[1]

    def word2features(self, sent, i):
        word = sent[i][0]
        postag = sent[i][1]
        token_seq = self.sent2tokens(sent[:i + 1])
        postag_seq = self.sent2postags(sent[:i + 1])

        # Common features for all words
        features = [
            'bias',
            'word.lower=' + word.lower(),
            'postag=' + postag,
            'wordperpl.isdecr=%s' % self.is_perplexity_decreasing(
                token_seq, self.tokens_lm[0], self.tokens_lm[1]),
            'postagperpl.isdecr=%s' % self.is_perplexity_decreasing(
                postag_seq, self.postags_lm[0], self.postags_lm[1])
        ]

        # Features for words that are not
        # at the beginning of a document
        if i > 0:
            word1 = sent[i-1][0]
            postag1 = sent[i-1][1]
            features.extend([
                '-1:word.lower=' + word1.lower(),
                '-1:postag=' + postag1
            ])
        else:
            # Indicate that it is the 'beginning of a document'
            features.append('BOS')

        # Features for words that are not
        # at the end of a document
        if i < len(sent)-1:
            word1 = sent[i+1][0]
            postag1 = sent[i+1][1]
            features.extend([
                '+1:word.lower=' + word1.lower(),
                '+1:postag=' + postag1
            ])
        else:
            # Indicate that it is the 'end of a document'
            features.append('EOS')

        return features

    def sent2features(self, sent):
        '''A function for extracting features from sentences'''
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent):
        '''A function for generating the output list of labels for each sentence'''
        return [label for (token, postag, label) in sent]

    def sent2postags(self, sent):
        '''A function for generating the output list of postags for each sentence'''
        if len(sent[0]) == 2:
            return [postag for (token, postag) in sent]  # for predicting sents
        return [postag for (token, postag, label) in sent]  # for training sents

    def sent2tokens(self, sent):
        '''A function for generating the output list of tokens for each sentence'''
        if len(sent[0]) == 2:
            return [token for (token, postag) in sent]  # for predicting sents
        return [token for (token, postag, label) in sent]  # for training sents

    def train(self):
        self.load_data()

        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',

            # coefficient for L1 penalty
            c1=0.5068521404310856,

            # coefficient for L2 penalty
            c2=0.024434096513563347,

            # maximum number of iterations
            max_iterations=200,

            # whether to include transitions that
            # are possible, but not observed
            all_possible_transitions=True
        )
        # Submit training data to the trainer
        crf.fit(self.X_train, self.y_train)

        # Save the model into 'crf.model' file
        self.save_model(crf)

    def test(self):

        # Generate predictions
        y_pred = self.crf.predict(self.X_test)

        # Random sample in the testing set
        i = 6
        for x, y in zip(y_pred[i], [x[1].split("=")[1] for x in self.X_test[i]]):
            print("%s (%s)" % (y, x))

        # Create a mapping of labels to indices
        labels = list(self.crf.classes_)

        # Print out the classification report
        print(metrics.flat_classification_report(
            self.y_test, y_pred, labels=labels, digits=3
        ))

    def save_model(self, crf):
        with open('crf.model', 'wb') as f:
            pickle.dump(crf, f)

    def load_model(self, name):
        with open(name, 'rb') as f:
            return pickle.load(f)

    def predict(self, tokens):
        '''Returns predicted labels from the text string'''
        X = self.sent2features(nltk.pos_tag(tokens))
        return self.crf.predict([X])[0]

    def segment_sent(self, s):
        '''Segments a sentence pased on prediction'''
        tokens = word_tokenize(s)
        y = self.predict(tokens)
        detokenizer = Detok()

        sents = []
        if 'P' in y:
            n = y.index('P') + 1
            sents.append(detokenizer.detokenize(tokens[:n]))
            sents.append(detokenizer.detokenize(tokens[n:]))
        else:
            sents.append(s)
        '''with MosesDetokenizer('en') as detokenize:
            if 'P' in y:
                n = y.index('P') + 1
                sents.append(detokenize(tokens[:n]))
                sents.append(detokenize(tokens[n:]))
            else:
                sents.append(s)'''

        return sents


if __name__ == '__main__':
    s = Segmenter()
    s.train()
    s.test()
