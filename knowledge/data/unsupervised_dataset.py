__author__ = 'Sun'

from knowledge.data.huge_frame import HugeFrame

class UnSupervisedDataSet(HugeFrame):

    def __init__(self, file_path,
                 feature_dim,
                 feature_info_file = None,
                 chunk_size = None,
                 compress = False,):

        super(UnSupervisedDataSet,self).__init__(
            file_path = file_path,
            frame_name= "data",
            feature_dim = feature_dim,
            feature_info_file = feature_info_file,
            chunk_size = chunk_size,
            compress = compress
        )


    def sample_batches(self, batch_size):

        for chunk in self.iter_chunks(chunk_size=batch_size):

            yield chunk.values

