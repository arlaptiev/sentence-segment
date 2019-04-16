import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.moses import MosesDetokenizer
from nltk.corpus import brown
import pycrfsuite
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class Segmenter:

    def __init__(self):
        '''Trains a CRF model'''
        data = self.get_sents()

        X = [self.extract_features(sent) for sent in data]
        y = [self.get_labels(doc) for doc in data]

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
    def extract_features(sent):
        '''A function for extracting featur es from sentences'''
        return [Segmenter.word2features(sent, i) for i in range(len(sent))]

    def get_labels(self, sent):
        '''A function for generating the output list of labels for each sentence'''
        return [label for (token, postag, label) in sent]

    def train(self):
        trainer = pycrfsuite.Trainer(verbose=True)

        # Submit training data to the trainer
        for xseq, yseq in zip(self.X_train, self.y_train):
            trainer.append(xseq, yseq)

        # Set the parameters of the model
        trainer.set_params({
            # coefficient for L1 penalty
            'c1': 0.1,

            # coefficient for L2 penalty
            'c2': 0.01,

            # maximum number of iterations
            'max_iterations': 200,

            # whether to include transitions that
            # are possible, but not observed
            'feature.possible_transitions': True
        })

        # Provide a file name as a parameter to the train function, such that
        # the model will be saved to the file when training is finished
        trainer.train('crf.model')

    def test(self):
        # Generate predictions
        tagger = pycrfsuite.Tagger()
        tagger.open('crf.model')
        y_pred = [tagger.tag(xseq) for xseq in self.X_test]

        # Let's take a look at a random sample in the testing set
        i = 6
        for x, y in zip(y_pred[i], [x[1].split("=")[1] for x in self.X_test[i]]):
            print("%s (%s)" % (y, x))

        # Create a mapping of labels to indices
        labels = {"S": 0, "P": 1}

        # Convert the sequences of tags into a 1-dimensional array
        predictions = np.array([labels[tag] for row in y_pred for tag in row])
        truths = np.array([labels[tag] for row in self.y_test for tag in row])

        # Print out the classification report
        print(classification_report(
            truths, predictions,
            target_names=["S", "P"]))

    @staticmethod
    def segment_sent(s):
        '''Segments a sentence pased on prediction'''
        tokens = word_tokenize(s)
        X = Segmenter.extract_features(nltk.pos_tag(tokens))
        tagger = pycrfsuite.Tagger()
        tagger.open('crf.model')
        y = tagger.tag(X)

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
