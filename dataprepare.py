import h5py
import pandas as pd
from datetime import datetime


# 特征加工
with h5py.File('D:/data_file/jqdata_copy.hdf5', 'r') as files:
    for key in list(files.keys())[1:]:
        try:
            for code in list(files[key].keys()):
                for _ in list(files[key][code].keys()):
                    # print(files[key][code][_]['columns'][:])
                    pass
        except RuntimeError:
            print("=======", key)
        assert 1 == 2
