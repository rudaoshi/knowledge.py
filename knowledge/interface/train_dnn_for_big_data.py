__author__ = 'Sun'


import ConfigParser
import click
import simplejson as json
import cPickle

from knowledge.data.supervised_dataset import SupervisedDataSet
from knowledge.machine.optimization.optim_factory import create_optimizer
from knowledge.machine.neuralnetwork.neuralnet_factory import create_neuralnet

from knowledge.util.theano_util import shared_dataset
@click.command()
@click.argument('config_file', type=click.Path(exists=True))
def train_dnn_for_big_data(config_file):

    config = ConfigParser.ConfigParser()
    config.read(config_file)


    input_sample_file = config.get("input", 'input_sample_file')
    data_group_name = config.get("input", 'data_group_name')

    output_model_file = config.get("output", 'output_model_file')

    network_setting = json.loads(config.get("network","architecture"))

    chunk_size = int(config.get("train", 'chunk_size'))
    optim_settings = json.loads(config.get("train", 'optim_settings'))

    train_data_set = SupervisedDataSet(input_sample_file,
                                       frame_name = data_group_name,
                                       chunk_size=chunk_size)

    optim_alog = create_optimizer(optim_settings)

    neural_net = create_neuralnet(network_setting)

    for (train_batch_x, train_batch_y) in train_data_set.sample_batches():
        shared_train_x, shared_train_y = shared_dataset((train_batch_x, train_batch_y))

        param = optim_alog.optimize(neural_net,
                                    neural_net.get_parameter(),
                                    shared_train_x,
                                    shared_train_y)

        neural_net.set_parameter(param)

    with open(output_model_file,'w') as output_file:

        cPickle.dump(neural_net.__get_states(),
            output_file,
            protocol=cPickle.HIGHEST_PROTOCOL)





