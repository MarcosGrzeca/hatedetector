# source: http://linanqiu.github.io/2015/10/07/word2vec-sentiment/

from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy

# random
from random import shuffle

# classifier
from sklearn.linear_model import LogisticRegression

size = 100


class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
    
    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
    
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences
    
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


def trainning():
    #./manage.py shell -c="from tweets.p2v import trainning; trainning()"

    sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS', 'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS', 'train-unsup.txt':'TRAIN_UNS'}


    sentences = LabeledLineSentence(sources)
    print("Let's train")

    model = Doc2Vec(min_count=1, window=10, vector_size=size, sample=1e-4, negative=5, workers=50)

    model.build_vocab(sentences.to_array())

    model.train(sentences.sentences_perm(), total_words=model.corpus_count, epochs=10)

    model.save('./imdb2.d2v')
    model = Doc2Vec.load('./imdb2.d2v')

    model['TRAIN_NEG_0']

    print("Let's train")
    
    train_arrays = numpy.zeros((25000, 100))
    train_labels = numpy.zeros(25000)

    for i in range(12500):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        train_arrays[i] = model[prefix_train_pos]
        train_arrays[12500 + i] = model[prefix_train_neg]
        train_labels[i] = 1
        train_labels[12500 + i] = 0

    test_arrays = numpy.zeros((25000, 100))
    test_labels = numpy.zeros(25000)

    for i in range(12500):
        prefix_test_pos = 'TEST_POS_' + str(i)
        prefix_test_neg = 'TEST_NEG_' + str(i)
        test_arrays[i] = model[prefix_test_pos]
        test_arrays[12500 + i] = model[prefix_test_neg]
        test_labels[i] = 1
        test_labels[12500 + i] = 0

    classifier = LogisticRegression()
    classifier.fit(train_arrays, train_labels)

    print(classifier.score(test_arrays, test_labels))



trainning()