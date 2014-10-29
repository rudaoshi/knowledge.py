__author__ = 'huang'

import cPickle
import gzip
import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T

def model_file_name(model_folder,model_name,param_name):
    prefix = os.path.join(model_folder,model_name)
    filename = prefix + '_' + param_name + '.npy'
    return filename

class BaseModule(object):

    def __init__(self,name):
        assert isinstance(name,str), 'Model name %s should be str type' % (str(name))
        self.name = name

    def params(self):
        # should return module's params in a list
        pass

    def save(self,model_folder):
        # save modules params to disk
        assert model_folder != '', 'MODULE FOLDER is empty'
        assert len(self.name) > 0, 'MODULE NAME is empty'
        prefix = os.path.join(model_folder,self.name)
        params = self.params()
        for p in params:
            filename = model_file_name(model_folder,self.name,p.name)
            val = p.get_value()
            np.save(filename,val)


    def load(self,model_folder):
        # load params from disk
        # sub class should implement this method
        assert len(self.name) > 0, 'MODULE NAME is empty'
        assert model_folder != '', 'MODULE FOLDER is empty'
        prefix = os.path.join(model_folder,self.name)
        params = self.params()
        self.params_lst = []
        for p in params:
            filename = model_file_name(model_folder,self.name,p.name)
            assert os.path.isfile(filename), 'MODULE %s PARAMS %s not exist' % (self.name,p.name)
            d = np.load(filename)
            self.params_lst.append(d)

class BaseModel(object):

    def __init__(self,name,model_folder=None):
        self.name = randomword(16)
        if model_folder == None:
            self.model_folder = os.environ.get('KG_model_FOLD','')
        else:
            self.model_folder = model_folder
        self.modules = []

    def add_module(self,module):
        self.modules.append(module)

    def save(self):
        for m in self.modules:
            m.save(self.model_folder)

    def load(self):
        # load model from disk
        assert len(self.name) > 0, 'MODULE NAME is empty'
        assert self.model_folder != '', 'MODEL FOLDER is empty'
        for m in self.modules:
            m.load(model_folder)
