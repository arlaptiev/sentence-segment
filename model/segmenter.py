import numpy as np
import pickle
import nltk
from nltk.tokenize import word_tokenize
from mosestokenizer import MosesDetokenizer
from nltk.corpus import brown
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split


class Segmenter:

    def __init__(self):
        '''Loads data for training and testing a CRF model'''
        data = self.get_sents()

        X = [self.sent2features(sent) for sent in data]
        y = [self.sent2labels(sent) for sent in data]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

    def label_sents(self, sents):
        '''Returns labeled sents: 'S' corresponding to a token right before a Space, and 'P' corresponding
        to a token right before a period'''
        #
        labeled_sents = []
        for s in sents:
            pairs = []
            # Label every token 'S' if a space goes after it or 'P' if a period follows
            for i in range(len(s) - 2):
                pairs.append([s[i], 'S'])

            pairs.append([s[len(s) - 2], 'P'])

            labeled_sents.append(pairs)
        return labeled_sents

    def get_sents(self):
        '''Returns joined, labeled, and POS tagged sentences from Brown corpus with no punctuation in between'''
        # Read data corpus into sents
        sents = brown.sents()
        labeled_sents = self.label_sents(sents)

        # Join every two sentence
        paragraphs = [a + b for a, b in zip(labeled_sents[::2], labeled_sents[1::2])]

        '''TODO for comma splice
        # Join every two sentence
        p1 = [a[:-1] + [[a[-1][0]] + ['S']] + [[',', 'P']] + b for a,
              b in zip(labeled_sents[::4], labeled_sents[1::4])]
        # Join every other two sentence with a comma after the first one - comma splice
        p2 = [a + b for a, b in zip(labeled_sents[2::4], labeled_sents[3::4])]

        paragraphs = p1 + p2'''

        data = []
        for p in paragraphs:
            # Delete the 'P' label from the end of the second sentence
            p[len(p) - 1][1] = 'S'

            # Obtain the list of tokens in the document
            tokens = [t for t, label in p]

            # Perform POS tagging
            tagged = nltk.pos_tag(tokens)

            # Take the word, POS tag, and its label
            data.append([(w, pos, label) for (w, label), (word, pos) in zip(p, tagged)])
        return data

    @staticmethod
    def word2features(sent, i):
        word = sent[i][0]
        postag = sent[i][1]

        # Common features for all words
        features = [
            'bias',
            'word.lower=' + word.lower(),
            'word[-3:]=' + word[-3:],
            'word[-2:]=' + word[-2:],
            'word.isupper=%s' % word.isupper(),
            'word.isdigit=%s' % word.isdigit(),
            'postag=' + postag
        ]

        # Features for words that are not
        # at the beginning of a document
        if i > 0:
            word1 = sent[i-1][0]
            postag1 = sent[i-1][1]
            features.extend([
                '-1:word.lower=' + word1.lower(),
                '-1:word.isupper=%s' % word1.isupper(),
                '-1:word.isdigit=%s' % word1.isdigit(),
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
                '+1:word.isupper=%s' % word1.isupper(),
                '+1:word.isdigit=%s' % word1.isdigit(),
                '+1:postag=' + postag1
            ])
        else:
            # Indicate that it is the 'end of a document'
            features.append('EOS')

        return features

    @staticmethod
    def sent2features(sent):
        '''A function for extracting features from sentences'''
        return [Segmenter.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent):
        '''A function for generating the output list of labels for each sentence'''
        return [label for (token, postag, label) in sent]

    def train(self):
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',

            # coefficient for L1 penalty
            c1=0.8377072127476861,

            # coefficient for L2 penalty
            c2=4.083015357819278e-05,

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
        # Load model
        crf = self.load_model()

        # Generate predictions
        y_pred = crf.predict(self.X_test)

        # Random sample in the testing set
        i = 6
        for x, y in zip(y_pred[i], [x[1].split("=")[1] for x in self.X_test[i]]):
            print("%s (%s)" % (y, x))

        # Create a mapping of labels to indices
        labels = list(crf.classes_)

        # Print out the classification report
        print(metrics.flat_classification_report(
            self.y_test, y_pred, labels=labels, digits=3
        ))

    def save_model(self, crf):
        with open('crf.model', 'wb') as f:
            pickle.dump(crf, f)

    @staticmethod
    def load_model():
        with open('crf.model', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def predict(tokens):
        '''Returns predicted labels of the string'''
        X = Segmenter.sent2features(nltk.pos_tag(tokens))
        return Segmenter.load_model().predict([X])[0]

    @staticmethod
    def segment_sent(s):
        '''Segments a sentence pased on prediction'''
        tokens = word_tokenize(s)
        y = Segmenter.predict(tokens)

        detokenizer = MosesDetokenizer()

        sents = []
        if 'P' in y:
            n = y.index('P') + 1
            sents.append(detokenizer.detokenize(tokens[:n], return_str=True))
            sents.append(detokenizer.detokenize(tokens[n:], return_str=True))
        else:
            sents.append(s)

        return sents


if __name__ == '__main__':
    s = Segmenter()
    s.train()
    s.test()
