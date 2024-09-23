import os


DATA_ROOT = os.path.join(os.path.dirname(__file__), '../data')


class DataCollection:
    def __init__(self, data, index_attr=None):
        self._data = data
        if index_attr is not None:
            self._indexedData = {
                str(instance[index_attr]): instance
                for instance in self._data
            }
        else:
            self._indexedData = {
                str(idx): instance
                for idx, instance in enumerate(self._data)
            }

    @property
    def data(self):
        return self._data

    def get(self, instance_id):
        return self._indexedData[instance_id]
