__author__ = 'sun'


__author__ = 'sunmingming01'

import click

from knowledge.data.huge_frame import HugeFrame

@click.command()
@click.argument('output_file', type=click.Path())
@click.argument('frame_name')
@click.argument('input_csv_file', type=click.Path(exists=True))
@click.argument('feature_dim', type=click.INT)
@click.option('--feature_desc_file', type=click.Path(exists=True))
@click.option('--separator')
@click.option('--compress', type=click.BOOL)
def make_huge_frame(output_file, frame_name,
                         input_csv_file,
                         feature_dim,
                         feature_desc_file = None,
                         separator = None,
                         compress = False):

    frame = HugeFrame(output_file, frame_name, compress = compress)

    if separator == "Space":
        separator = " "
    elif separator == "Tab":
        separator = "\t"
    else:
        separator = separator

    frame.append_data_from_txt(input_csv_file, feature_dim, feature_desc_file, sep = separator)

if __name__ == "__main__":

    make_huge_frame()
