__author__ = 'Sun'

from knowledge.data.huge_frame import HugeFrame

class SupervisedDataSet(HugeFrame):

    def __init__(self, file_path, frame_name = 'data',
                 chunk_size = None,
                 compress = False,):

        super(SupervisedDataSet,self).__init__(
            file_path = file_path,
            frame_name= frame_name,
            chunk_size = chunk_size,
            compress = compress
        )


    def sample_batches(self, batch_size = None):

        for chunk in self.iter_chunks(chunk_size=batch_size):
            target = chunk.ix[:, 0]
            feature = chunk.ix[:, 1:]

            yield (feature.values, target.values)

