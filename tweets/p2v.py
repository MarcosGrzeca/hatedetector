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

    sources = {'test-pos.txt':'TEST_YES', 'test-neg.txt':'TEST_NO', 'train-pos.txt':'TRAIN_YES', 'train-neg.txt':'TRAIN_NO', 'train-unsup.txt':'TRAIN_UNS'}

    sentences = LabeledLineSentence(sources)
    print("Let's train")

    # model = Doc2Vec(min_count=1, window=10, vector_size=size, sample=1e-4, negative=5, workers=500)

    # model.build_vocab(sentences.to_array())

    # model.train(sentences.sentences_perm(), total_words=model.corpus_count, epochs=10)

    # model.save('kagggle.d2v')

    # print("Final model")
    # test()



def test():
    #./manage.py shell -c="from tweets.p2v import test; test()"

    model = Doc2Vec.load('kagggle.d2v')

    tamTrainNo = 12500
    tamTrainYes = 12500

    tamTestNo = 12500
    tamTestYes = 12500

    train_arrays = numpy.zeros((tamTrainNo + tamTrainYes, size))
    train_labels = numpy.zeros(tamTrainNo + tamTrainYes)

    for i in range(tamTrainNo):
        train_arrays[i] = model.docvecs['TRAIN_NO_' + str(i)]
        train_labels[i] = 0

    for i in range(tamTrainYes):
        train_arrays[tamTrainNo+i] = model.docvecs['TRAIN_YES_' + str(i)]
        train_labels[tamTrainNo+i] = 1

    test_arrays = numpy.zeros((tamTestNo + tamTestYes, size))
    test_labels = numpy.zeros(tamTestNo + tamTestYes)

    for i in range(tamTestNo):
        test_arrays[i] = model.docvecs['TEST_NO_' + str(i)]
        test_labels[i] = 0

    for i in range(tamTestYes):
        test_arrays[tamTestNo+i] = model.docvecs['TEST_YES_' + str(i)]
        test_labels[tamTestNo+i] = 1

    #print(train_labels)
    #print(test_labels)

    #numpy.savetxt("kagggle_train_LogisticRegression_labels.csv", numpy.asarray(train_labels), delimiter=",")
    #numpy.savetxt("kagggle_teste_LogisticRegression_labels.csv", numpy.asarray(test_labels), delimiter=",")

    classifier = LogisticRegression()
    classifier.fit(train_arrays, train_labels)
    print(classifier.score(test_arrays, test_labels))

trainning()