def word2features1(sent, i):
    '''
    5000 training samples, 100 max_iterations
              precision    recall  f1-score   support

           S       0.98      1.00      0.99     19810
           P       0.71      0.32      0.44       500

avg / total       0.97      0.98      0.97     21416
    '''
    word = sent[i][0]
    postag = sent[i][1]

    seq = ''
    start = i-5 if i > 4 else 0
    for j in range(start, i):
        seq += '-' + sent[j][1]

    # Common features for all words
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'sequence=' + seq,
        'postag=' + postag
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


def word2features2(sent, i):
    '''
    5000 training samples, 100 max_iterations
              precision    recall  f1-score   support

           S       0.98      0.99      0.99     20735
           P       0.57      0.33      0.42       500

avg / total       0.97      0.98      0.97     21416
    '''
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


def word2features(sent, i):
    return word2features2(sent, i)
