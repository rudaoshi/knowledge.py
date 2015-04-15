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
def train_dnn_for_big_data(config_file):

    config = ConfigParser.ConfigParser()
    config.read(config_file)

    hadoop_bin = config.get("hadoop", 'bin')

    temp_dir = config.get('temp','temp_dir')

    sample_list = config.get("input", 'sample_file_list')
    frame_name = config.get("input", 'data_frame_name')

    output_model_prefix = config.get("output", 'output_model_prefix')

    try:
        network_arch = json.loads(config.get("network","architecture"))
    except:
        print config.get("network","architecture")
        raise

    max_epoches = int(config.get("train", 'max_epoches'))
    chunk_size = int(config.get("train", 'chunk_size'))
    optim_settings = json.loads(config.get("train", 'optim_settings'))

    neuralnet = create_neuralnet(network_arch)
    optimizer = create_optimizer(optim_settings)

    neuralnet.prepare_learning(optimizer.get_batch_size())

    for i in range(max_epoches):

        shuffle(sample_list)
        for remote_file_path in sample_list:

            local_file_path = download_file(hadoop_bin, remote_file_path, temp_dir)

            train_data_set = SupervisedDataSet(local_file_path, frame_name=frame_name)

            print time.ctime() + ":\tbegin training with sample : " + remote_file_path

            print time.ctime() + ":\tbegin epoche :", i
            for idx, (train_X, train_y_) in enumerate(train_data_set.sample_batches(batch_size=chunk_size)):

                print time.ctime() + ":\tbegin new chunk : ", idx, "@epoch : ", i
                train_y = numpy.zeros((train_y_.shape[0], 1))
                train_y[:,0] = train_y_
                neuralnet.update_chunk(train_X, train_y)

                new_param = optimizer.optimize(neuralnet, neuralnet.get_parameter())

                neuralnet.set_parameter(new_param)

            os.system('rm ' + local_file_path)


            with open(output_model_prefix + "_"  + str(i) + "_" +
                                   os.path.basename(local_file_path), 'w') as f:
                content = network_arch
                content["parameter"] = neuralnet.get_parameter()
                cPickle.dump(content, f, protocol=cPickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    train_dnn_for_big_data()