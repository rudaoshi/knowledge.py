__author__ = 'Sun'


import ConfigParser
import click
import simplejson as json

from knowledge.machine.neuralnetwork.neuralnet_factory import create_neuralnet
from knowledge.machine.optimization.optim_factory import create_optimizer
from knowledge.util.theano_util import shared_dataset

from knowledge.data.supervised_dataset import SupervisedDataSet

import cPickle

@click.command()
@click.argument('config_file', type=click.Path(exists=True))
def train_dnn_for_big_data(config_file):

    config = ConfigParser.ConfigParser()
    config.read(config_file)

    input_sample_file = config.get("input", 'input_sample_file')
    frame_name = config.get("input", 'data_frame_name')

    output_model_file = config.get("output", 'output_model_file')

    try:
        network_arch = json.loads(config.get("network","architecture"))
    except:
        print config.get("network","architecture")
        raise

    chunk_size = int(config.get("train", 'chunk_size'))
    optim_settings = json.loads(config.get("train", 'optim_settings'))

    neuralnet = create_neuralnet(network_arch)
    optimizer = create_optimizer(optim_settings)

    train_data_set = SupervisedDataSet(input_sample_file,frame_name=frame_name)

    for train_X, train_y in train_data_set.sample_batches(batch_size=chunk_size):

        shared_train_X, shared_train_y = shared_dataset((train_X, train_y))

        new_param = optimizer.optimize(neuralnet, neuralnet.get_parameter(), shared_train_X, shared_train_y)

        neuralnet.set_parameter(new_param)


    with open(output_model_file, 'w') as f:
        content = network_arch
        content["parameter"] = neuralnet.get_parameter()
        cPickle.dump(content, f, protocol=cPickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    train_dnn_for_big_data()