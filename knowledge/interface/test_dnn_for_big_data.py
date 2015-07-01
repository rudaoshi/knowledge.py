__author__ = 'Sun'


import ConfigParser
import click
import simplejson as json

from knowledge.machine.neuralnetwork.neuralnet_factory import create_neuralnet
from knowledge.machine.optimization.optim_factory import create_optimizer
from knowledge.util.hadoop import download_file
from knowledge.data.supervised_dataset import SupervisedDataSet
import numpy
import cPickle
import time
import os
from random import shuffle


@click.command()
@click.argument('config_file', type=click.Path(exists=True))
def test_dnn_for_big_data(config_file):

    config = ConfigParser.ConfigParser()
    config.read(config_file)

    hadoop_bin = config.get("hadoop", 'bin')

    temp_dir = config.get('temp','temp_dir')

    sample_file_list = config.get("input", 'sample_file_list')
    frame_name = config.get("input", 'data_frame_name')
    chunk_size = int(config.get("input", 'chunk_size'))

    model_file_path = config.get("model", 'model_file_path')

    predict_file_path = config.get("output", 'predict_file_path')

    with open(model_file_path, 'r') as model_file:

        model_data = cPickle.load(model_file)

    parameter = model_data["parameter"]
    del model_data["parameter"]

    neuralnet = create_neuralnet(model_data)
    neuralnet.set_parameter(parameter)


    sample_file_paths = []
    with open(sample_file_list,'r') as f:
        for line in f:
            line = line.strip()
            if line:
                sample_file_paths.append(line)

    predict_file = open(predict_file_path, 'w')

    for file_path in sample_file_paths:

        if file_path.startswith("hdfs:"):
            local_file_path = download_file(hadoop_bin, file_path, temp_dir)
        else:
            local_file_path = file_path
        train_data_set = SupervisedDataSet(local_file_path, frame_name=frame_name)

        print time.ctime() + ":\tbegin predict with sample : " + remote_file_path

        for idx, (train_X, train_y_) in enumerate(train_data_set.sample_batches(batch_size=chunk_size)):

            predict_y = neuralnet.predict(train_X)

            output_val = numpy.concatenate((train_y_, predict_y), axis=1)
            predict_file.write("\n".join("\t".join(x) for x in output_val))
            predict_file.write("\n")

        if file_path.startswith("hdfs:"):
                os.system('rm ' + local_file_path)
    predict_file.close()


if __name__ == "__main__":

    test_dnn_for_big_data()