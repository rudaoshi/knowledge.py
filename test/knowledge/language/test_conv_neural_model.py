import theano
import theano.tensor as T
import numpy as np
import pprint
import time
import sys
import os

from knowledge.machine.neuralnetwork.layer.mlp import HiddenLayer
from knowledge.machine.neuralnetwork.layer.logistic_sgd import LogisticRegression
from knowledge.machine.neuralnetwork.layer.conv_layer import SrlConvLayer
from knowledge.machine.neuralnetwork.layer.lookup_table_layer import LookupTableLayer
from knowledge.util.theano_util import shared_dataset

from knowledge.language.core.corpora import Conll05
from knowledge.language.neural_model.problem.srl_problem import SrlProblem
from knowledge.language.neural_model.conv_neural_model import SrlNeuralLanguageModel
from knowledge.language.core.definition import SrlTypes
from knowledge.language.core.corpora import Corpora
from knowledge.language.core.word import Word

def xtest_foo():
    mini_batch_size = 1000
    word_num = 10000
    sent_size = 30
    word_feature_num = 20
    wordpos_feature_num = 20
    window_size = 5

    conv_hidden_feature_num = 3
    conv_window = 7

    rng = np.random.RandomState(1234)
    inputs = T.itensor3('inputs')

    # inputs shpae: (batch_size,max_sentence_length+3,max_sentence_length)
    wordvect = LookupTableLayer(inputs = inputs[:,0:1,:], table_size = word_num,
            window_size = window_size, feature_num = word_feature_num,
            reshp = (inputs.shape[0],1,1,inputs.shape[2]*word_feature_num))


    wordpos_vect = LookupTableLayer(inputs = inputs[:,3:,:], table_size = word_num,
           window_size = window_size, feature_num = wordpos_feature_num,
           reshp = (inputs.shape[0],inputs.shape[1]-3,1,inputs.shape[2]*wordpos_feature_num))


    dinputs = np.random.randint(low=0,high=19,size=(mini_batch_size,sent_size+3,sent_size)).astype(np.int32)
    foo1a = theano.function(inputs=[inputs],outputs=wordvect.output)
    outputs1a = foo1a(dinputs)
    print 'test lookup, outputs1a shape',outputs1a.shape
    foo1b = theano.function(inputs=[inputs],outputs=wordpos_vect.output)
    outputs1b = foo1b(dinputs)
    print 'test lookup, outputs1b shape',outputs1b.shape

    conv_word = SrlConvLayer('conv_word',rng,wordvect.output,\
            conv_hidden_feature_num,1,conv_window,word_feature_num)
    foo2a = theano.function(inputs=[inputs],outputs = conv_word.out)
    outputs2a = foo2a(dinputs)
    print 'test conv1, outputs2a shape',outputs2a.shape
    conv_wordpos = SrlConvLayer('conv_wordpos',rng,wordpos_vect.output,\
            conv_hidden_feature_num,sent_size,conv_window,wordpos_feature_num)
    foo2b = theano.function(inputs=[inputs],outputs = conv_wordpos.output)
    outputs2b = foo2b(dinputs)
    print 'test conv1, outputs2b shape',outputs2b.shape

    conv_out = conv_word.out  + conv_wordpos.output
    foo3a = theano.function(inputs=[inputs],outputs = conv_out)
    outputs3a = foo3a(dinputs)
    print 'test conv, outputs3a shape',outputs3a.shape
    print 'test_foo done'

def test_srl_conv_network():
    print '*' * 20
    print 'test_srl_conv_network'
    home = os.path.expanduser('~')
    filename = os.path.join(home,'data/conll05/training-set')
    raw_corpora = Conll05.loadraw(filename)
    srl_sents = []
    for sent in raw_corpora:
        iobsent = Conll05.sentence2iobsentece(sent)
        srl_sents += iobsent

    words = Corpora(pading_lst=[Word.padding_word(),Word.padding_word2()])
    pos = Corpora(pading_lst=[Word.padding_pos(),Word.padding_pos2()])
    srl_problem = SrlProblem(words,pos,srl_sents)
    max_term_per_sent = 141
    window_size = 11
    pos_conv_size = 15
    max_size = max_term_per_sent + window_size - 1
    print 'corpora has words',len(Conll05.words)
    print 'corpora has pos',len(Conll05.pos)
    print 'corpora has tags',len(Conll05.tags)
    print 'window_size' , window_size
    print 'pos_conv_size', pos_conv_size
    print 'max_term_per_sent', max_term_per_sent
    print 'max_size',max_size


    validation_frequency = 100

    model_params = dict()
    model_params['L1_reg'] = 0.0
    model_params['L2_reg'] = 0.1

    # max_sentence_length = max_term_per_sent + window_size - 1
    # which is the maximum length of each sentence with padding
    model_params['max_sentence_length'] = max_size
    # which is also conv window size
    model_params['window_size'] = window_size
    model_params['word_num'] = len(Conll05.words)
    model_params['POS_num'] = len(Conll05.pos)
    # how many pos should we consider in model
    model_params['verbpos_num'] = window_size + 1
    model_params['wordpos_num'] = window_size + 1
    model_params['position_conv_half_window'] = (window_size - 1) / 2

    # the dimension of word vector
    model_params['word_feature_num'] = 30
    # the dimension of POS vector
    model_params['POS_feature_num'] = 30
    # the dimension of word's position vector
    model_params['wordpos_feature_num'] = 30
    # the dimension of verb's position vector
    model_params['verbpos_feature_num'] = 30

    model_params['conv_window'] = window_size
    model_params['conv_hidden_feature_num'] = 20

    model_params['hidden_layer_size'] = 100
    model_params['tags_num'] = len(Conll05.tags)


    print 'model params'
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(model_params)

    rng = np.random.RandomState(1234)
    model = SrlNeuralLanguageModel(rng,model_params)

    # exp params
    n_epochs = 1000
    batch_iter_num = 10
    learning_rate = 0.001
    validation_frequency = 100

    epoch = 0
    done_looping = False
    while (epoch < n_epochs) and (not done_looping):
        iter = 0
        for idx,data in enumerate(srl_problem.get_batch(batch_size = 100000,
            window_size = window_size,
            max_size = max_size)):
            X,Y,sent_len,masks = data
            print 'data %d, X shape %s,Y shape %s,sent_len shape %s,masks shape %s' % (idx,str(X.shape),str(Y.shape),str(sent_len.shape),str(masks.shape))
            X = X.astype(np.int32)
            Y = Y.astype(np.int32)
            sent_len = sent_len.astype(np.int32)
            masks = masks.astype(np.int32)

            '''
            if iter % validation_frequency == 0:
                error,time_cost = model.valid(X,Y,sent_len,masks)
                print >> sys.stderr, 'epoch %i, minibatch %i/%i, validation error %f %%,cost time %f' % \
                     (epoch, iter,100,this_validation_loss * 100.,time_cost)
                pass
            else:
                minibatch_avg_cost,time_cost = model.fit_batch(X,Y,sent_len,masks)
                print >> sys.stderr, 'epoch %i, minibatch %i/%i, minibatch cost,cost time %f', \
                        (epoch,iter,100,minibatch_avg_cost,time_cost)
            '''
            minibatch_avg_cost,time_cost = model.test_foo(X,Y,sent_len,masks)
            print 'time cost',time_cost
            iter += 1
        epoch += 1


