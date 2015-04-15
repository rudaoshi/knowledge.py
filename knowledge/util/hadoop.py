__author__ = 'Sun'


import os
def download_file(hadoop_bin, file_path, save_dir):

    return_code = os.system(" ".join([hadoop_bin, " fs -get ", file_path, save_dir]))

    if return_code != 0:
        raise Exception("Cannot download file " + file_path)

    downloaded_file_path = os.path.join(save_dir, os.path.basename(file_path))

    return downloaded_file_path