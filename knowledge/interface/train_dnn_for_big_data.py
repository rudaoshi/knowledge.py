__author__ = 'Sun'


import ConfigParser
import click
import simplejson as json

@click.command()
@click.argument('config_file', type=click.Path(exists=True))
def train_dnn_for_big_data(config_file):

    config = ConfigParser.ConfigParser()
    config.read(config_file)


    input_sample_file = config.get("input", 'input_sample_file')
    data_group_name = config.get("input", 'data_group_name')

    output_model_file = config.get("output", 'output_model_file')

    network_arch = json.loads(config.get("network","architecture"))

    chunk_size = int(config.get("train", 'chunk_size'))
    optim_algo = config.get("train", 'optim_algo')
    optim_settings = json.loads(config.get("train", 'optim_settings'))
