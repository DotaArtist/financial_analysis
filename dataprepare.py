import h5py
import pandas as pd
from datetime import datetime


# 特征加工
with h5py.File('D:/data_file/jqdata_bak_20210223.hdf5', 'r') as files:
    for key in list(files.keys()):
        for code in list(files[key].keys()):
            tmp_list = []

            for _ in list(files[key][code].keys()):
                columns = files[key][code][_]['columns'][:]
                columns = ['{}@{}'.format(_, i.decode('utf-8')) for i in columns]

                values = files[key][code][_]['values'][:]

                tmp_df = pd.DataFrame(values)
                tmp_df.columns = columns
                tmp_list.append(tmp_df)

            assert 1==2
