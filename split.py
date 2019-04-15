import codecs
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import brown
import pycrfsuite
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import features

# Read data corpus into sents
sents = brown.sents()

# Part 2
labeled_sents = []
for s in sents:
    pairs = []

    # Label every token 'S' if a space goes after it or 'P' if a period follows
    for i in range(len(s) - 2):
        pairs.append([s[i], 'S'])

    pairs.append([s[len(s) - 2], 'P'])

    labeled_sents.append(pairs)

paragraphs = [a + b for a, b in zip(labeled_sents[::2], labeled_sents[1::2])]

for p in paragraphs:
    p[len(p) - 1][1] = 'S'


# Part 3

data = []
for p in paragraphs:

    # Obtain the list of tokens in the document
    tokens = [t for t, label in p]

    # Perform POS tagging
    tagged = nltk.pos_tag(tokens)

    # Take the word, POS tag, and its label
    data.append([(w, pos, label) for (w, label), (word, pos) in zip(p, tagged)])

# Part 4


def word2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]

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
        word1 = doc[i-1][0]
        postag1 = doc[i-1][1]
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
    if i < len(doc)-1:
        word1 = doc[i+1][0]
        postag1 = doc[i+1][1]
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


def extract_features(sent):
    '''A function for extracting featur es from sentences'''
    return [word2features(sent, i) for i in range(len(sent))]


def get_labels(sent):
    '''A function for generating the output list of labels for each sentence'''
    return [label for (token, postag, label) in sent]

# Part 5


X = [extract_features(sent) for sent in data]
y = [get_labels(doc) for doc in data]

print(paragraphs)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def train():
    # Part 6

    trainer = pycrfsuite.Trainer(verbose=True)

    # Submit training data to the trainer
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    # Part 7

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

    # Part 8

    # Provide a file name as a parameter to the train function, such that
    # the model will be saved to the file when training is finished
    trainer.train('crf.model')


def test():
    # Part 9

    # Generate predictions
    tagger = pycrfsuite.Tagger()
    tagger.open('crf.model')
    y_pred = [tagger.tag(xseq) for xseq in X_test]

    # Let's take a look at a random sample in the testing set
    i = 6
    for x, y in zip(y_pred[i], [x[1].split("=")[1] for x in X_test[i]]):
        print("%s (%s)" % (y, x))

    # Part 10

    # Create a mapping of labels to indices
    labels = {"S": 0, "P": 1}

    # Convert the sequences of tags into a 1-dimensional array
    predictions = np.array([labels[tag] for row in y_pred for tag in row])
    truths = np.array([labels[tag] for row in y_test for tag in row])

    # Print out the classification report
    print(classification_report(
        truths, predictions,
        target_names=["S", "P"]))

    print(accuracy_score(truths, predictions))


train()
test()
