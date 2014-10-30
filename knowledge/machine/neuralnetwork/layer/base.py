__author__ = 'huang'

import cPickle
import gzip
import os
import sys
import time
import cPickle
import datetime

import numpy as np

import theano
import theano.tensor as T

def model_file_name(model_folder,model_name,tag=None,add_time_stamp=False):
    fmt='%y%m%d-%H:%M:%S'
    prefix = os.path.join(model_folder,model_name)
    if tag !=None and len(tag) > 0:
        prefix += '_' + tag
    if add_time_stamp:
        prefix += '_' + datetime.datetime.now().strftime(fmt)
    filename = prefix + '.model'
    return filename

class BaseModel(object):

    def __init__(self,name,model_folder=None):
        assert isinstance(name,str) and len(name) > 0 , 'Model\'s name should be string with at least one character'
        assert isinstance(model_folder,str) and len(model_folder) > 0 , 'MODEL FOLDER is empty'
        self.name = name
        if model_folder == None:
            self.model_folder = os.environ.get('KG_MODEL_FOLD','')
        else:
            self.model_folder = model_folder

    def dump_core(self,tag=None,add_time_stamp=False):
        filename = model_file_name(self.model_folder,self.name,tag,add_time_stamp)
        with open(filename,'wb') as fw:
            cPickle.dump(self.core, fw, protocol=cPickle.HIGHEST_PROTOCOL)

    def load_core(self,model_file):
        # load model from disk
        assert isinstance(model_file,str) and len(model_file) > 0, 'MODULE NAME is empty'
        filename = os.path.join(self.model_folder,model_file)
        with open(filename,'rb') as fr:
            return cPickle.load(fr)
