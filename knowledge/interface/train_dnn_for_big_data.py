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

    sample_file_list = config.get("input", 'sample_file_list')
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

    optimizer.work_for(neuralnet)

    sample_file_paths = []
    with open(sample_file_list,'r') as f:
        for line in f:
            line = line.strip()
            if line:
                sample_file_paths.append(line)


    for i in range(max_epoches):
        print time.ctime() + ":\tbegin epoche :", i
        shuffle(sample_file_paths)
        for file_path in sample_file_paths:

            if file_path.startswith("hdfs:"):
                local_file_path = download_file(hadoop_bin, file_path, temp_dir)
            else:
                local_file_path = file_path

            train_data_set = SupervisedDataSet(local_file_path, frame_name=frame_name)

            print time.ctime() + ":\tbegin training with sample : " + file_path

            try:

                for idx, (train_X, train_y) in enumerate(train_data_set.sample_batches(batch_size=chunk_size)):

                    print time.ctime() + ":\tbegin new chunk : ", idx, "@epoch : ", i

                    optimizer.update_chunk(train_X, train_y)

                    new_param = optimizer.optimize(None)#neuralnet.get_parameter())

#                    neuralnet.set_parameter(new_param)
            except Exception as e:

                print e.message


            if file_path.startswith("hdfs:"):
                os.system('rm ' + local_file_path)


        with open(output_model_prefix + "_"  + str(i) + ".dat", 'w') as f:
            content = network_arch
            content["parameter"] = neuralnet.get_parameter()
            cPickle.dump(content, f, protocol=cPickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    train_dnn_for_big_data()