__author__ = 'sunmingming01'

import csv
from collections import defaultdict
import collections

import pandas as pd
import numpy as np
from sklearn.externals import joblib

from knowledge.util.math.stat import roulette
from knowledge.data.data_discription import make_dtype_from_str
#from toolchain.mlbase.data.acc import to_matrix_format_str
#from toolchain.mlbase.data.scaler import MinMaxScaler


class FrameStore(object):

    def __init__(self, file_path, frame_name, compress = False):

        self.file_path = file_path
        self.frame_name = frame_name
        self.compress = compress

        if self.compress:
            self.store = pd.HDFStore(self.file_path, complevel=1, complib='blosc')
        else:
            self.store = pd.HDFStore(self.file_path)

    def __enter__(self):

        return self

    def __exit__(self, type, value, traceback):

        self.store.close()

    def append(self, chunk):

        self.store.append(self.frame_name, chunk)

    def close(self):

        self.store.close()

class HugeFrame(object):


    def __init__(self, file_path, frame_name,
                 chunk_size = None,
                 compress = False,):

        self.__file_path = file_path
        self.__frame_name = frame_name

        self.chunk_size = chunk_size
        if not self.chunk_size:
            self.chunk_size = 50000

        self.compress = compress


    def append_data_from_txt(self, txt_feature_file_path, feature_dim,
                 feature_info_file = None, **kwargs):

        if not feature_info_file:
            dtype = np.dtype([("", np.float32)] * feature_dim)

        else:
            with open(feature_info_file, 'r') as f:
                info_str = f.read().strip()
                dtype = make_dtype_from_str(info_str)

                assert len(dtype.names) == feature_dim, "Feature dim is not equal to dtype size"

        reader = pd.read_table(txt_feature_file_path,
                               names=dtype.names,
                               dtype=dtype,
                               quoting=csv.QUOTE_NONE,
                               chunksize=self.chunk_size,
                               **kwargs)

        store = pd.HDFStore(self.__file_path)
        for chunk in reader:
            chunk = chunk.dropna(axis=0)
            store.append(self.__frame_name, chunk, chunksize=self.chunk_size)
        store.close()


    def merge(self, other_frame):

        with self.open_storer() as store:
            for chunk in other_frame.iter_chunks():

                store.append(chunk)

    def read_columns(self, columns):
        reader = pd.read_hdf(self.__file_path,self.__frame_name, chunksize = self.chunk_size, columns=columns)

        data = []
        for chunk in reader:
            data.append(chunk)

        return pd.concat(data)



    def iter_chunks(self, columns = None, chunk_size = None ):

        if not chunk_size:
            chunk_size = self.chunk_size

        reader = pd.read_hdf(self.__file_path,
                             self.__frame_name,
                             chunksize = chunk_size,
                             columns = columns)

        for chunk in reader:
            yield chunk


    def sampling_by_column(self, sampled_frames, unsampled_frame, column_name, ratios):
        '''
        Split the data set by sampling query groups according to given raito
        :param ratios: Sampling ratios
        :return: yield group
        '''

        assert len(ratios) == len(sampled_frames), "The number of ratios do not match that of data paths"

#        output_file = tempfile.NamedTemporaryFile(delete=False)

        cumsum_ratios = np.cumsum(ratios)
        selected_idx = -1
        query_list = []
        for chunk in self.iter_chunks(columns = [column_name]):
            query_list.extend(list(chunk[column_name]))

        selected_group = defaultdict(set)

        old_query = ""
        for idx, query in enumerate(query_list):

            if query != old_query:
                selected_idx = roulette(cumsum_ratios)

            selected_group[selected_idx].add(idx)

            old_query = query


        chunksize = self.chunk_size
        for idx, chunk in enumerate(self.iter_chunks()):
            for data_id in range(len(ratios)):
                cur_idx = [i - idx*chunksize for i in range(idx*chunksize, (idx+1)*chunksize)
                           if i in selected_group[data_id]]
                filterd_chunk = chunk.take(cur_idx)

                sampled_frames[data_id].append(filterd_chunk, min_itemsize = 512)

            unsampled_idx = [i - idx*chunksize for i in range(idx*chunksize, (idx+1)*chunksize)
                           if i in selected_group[-1]]
            unsampled_chunk = chunk.take(unsampled_idx)
            unsampled_frame.append(unsampled_chunk, min_itemsize = 512)



    def __transform_by_expression(self, output_frame, expresses):
        """
        Transform the feature set
        :param expresses: list of strings that describe the transform. ex: d = b * e
        :return: yield transformed blocks
        """

        for chunk in self.iter_chunks():
            for transform in expresses:
                chunk.eval(transform)
                changed_column = transform.split('=')[0].strip()
                chunk[changed_column] = chunk[changed_column].astype('float32')

            output_frame.append(chunk)

    def __transform_by_model(self, output_frame, transformer):
        """
        Transform the feature set
        :param transforms: list of strings that describe the transform. ex: d = b * e
        :return: yield transformed blocks
        """

        for chunk in self.iter_chunks():

            output_frame.append(transformer.transform(chunk))


    def transform(self, output_frame, x):

        if isinstance(x, collections.Iterable):
            self.__transform_by_expression(output_frame, x)
        else:
            self.__transform_by_model(output_frame, x)

    # def get_min_max_scaler(self, feature_columns, scaler_path):
    #     """
    #     Transform the feature set
    #     :param transforms: list of strings that describe the transform. ex: d = b * e
    #     :return: yield transformed blocks
    #     """
    #
    #     global_max = np.ones((len(feature_columns),), dtype=np.float32) * np.float32(-np.inf)
    #     global_min = np.ones((len(feature_columns),), dtype=np.float32) * np.float32( np.inf)
    #     for chunk in self.iter_chunks(columns=feature_columns):
    #         concerned = chunk.values
    #         max_val = concerned.max(axis = 0)
    #         min_val = concerned.min(axis = 0)
    #
    #         global_max = np.maximum(max_val, global_max)
    #         global_min = np.minimum(min_val, global_min)
    #
    #     scaler = MinMaxScaler(feature_columns, global_min, global_max)
    #     joblib.dump(scaler, scaler_path)
    #
    #
    # def scale(self, output_frame, scaler_path):
    #
    #     scaler = joblib.load(scaler_path)
    #
    #     for chunk in self.iter_chunks():
    #         chunk = scaler.scale(chunk)
    #
    #         output_frame.append(chunk, min_itemsize = 512)
    #


    # def export_ranksvm_file(self, output_file_path, label_column, query_column, feature_colums):
    #     """
    #     Transform the feature set
    #     :param transforms: list of strings that describe the transform. ex: d = b * e
    #     :return: yield transformed blocks
    #     """
    #
    #     output_file = open(output_file_path, 'w')
    #
    #     needed_columns = [label_column, query_column] + feature_colums
    #     for chunk in self.iter_chunks(columns=needed_columns):
    #         queries = list(chunk[query_column])
    #         qids = np.array([QIDMap.Instance().get_qid(query) for query in queries])
    #
    #         string = to_ranksvm_format_str(chunk[label_column].values, qids, chunk[feature_colums].values)
    #         #
    #         # chunk['qid'] = qids
    #
    #         output_file.write(string)
    #
    #     output_file.close()



    def eval(self, expression):
        """
        Transform the feature set
        :param transforms: list of strings that describe the transform. ex: d = b * e
        :return: yield transformed blocks
        """

        output = []
        for chunk in self.iter_chunks():
            value = chunk.eval(expression)

            output.extend(value)

        return np.array(output)



    # def dump_columns(self, output_file_path,  column_names):
    #     """
    #     Transform the feature set
    #     :param transforms: list of strings that describe the transform. ex: d = b * e
    #     :return: yield transformed blocks
    #     """
    #
    #     output_file = open(output_file_path, 'w')
    #
    #     for chunk in self.iter_chunks(columns=column_names):
    #
    #         string = to_matrix_format_str(chunk.values)
    #
    #         output_file.write(string)
    #
    #     output_file.close()

    def dropna(self, output_frame):
        """
        Transform the feature set
        :param transforms: list of strings that describe the transform. ex: d = b * e
        :return: yield transformed blocks
        """
        for chunk in self.iter_chunks():
            chunk = chunk.dropna(axis=0)

            output_frame.append(chunk, min_itemsize = 512)

