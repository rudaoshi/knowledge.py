__author__ = 'Sun'

from knowledge.data.huge_frame import HugeFrame

class SupervisedDataSet(HugeFrame):

    def __init__(self, file_path, frame_name = 'data',
                 target_column_name = None,
                 chunk_size = None,
                 compress = False,):

        super(SupervisedDataSet,self).__init__(
            file_path = file_path,
            frame_name= frame_name,
            chunk_size = chunk_size,
            compress = compress
        )

        with self.open_storer() as store:
            dtype = store.dtypes

        self.target_column_name = target_column_name

        if not self.target_column_name:
            self.target_column_name = dtype.index[0]

        self.feature_columns = [feature_name for feature_name in dtype.index
                        if feature_name != self.target_column_name]

    # def get_sample_num(self):
    #
    #     with self.open_storer() as store:
    #         return store.__len__()

    def get_dimension(self):

        with self.open_storer() as store:
            return len(store.dtypes.index) - 1


    def sample_batches(self, batch_size = None):

        for chunk in self.iter_chunks(chunk_size=batch_size):
            target = chunk[self.target_column_name]
            feature = chunk[self.feature_columns]

            yield (feature.values, target.values)

