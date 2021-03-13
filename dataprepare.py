import h5py
import pandas as pd
from datetime import datetime
import gc


# 特征加工
with h5py.File('D:/data_file/jqdata_bak_20210223.hdf5', 'r') as files:

    df_daily_list = []
    for key in list(files.keys()):
        df_list = []  # 每日数据

        for code in list(files[key].keys()):
            tmp_list = []

            for _ in list(files[key][code].keys()):
                if _ != "price_minute":
                    columns = files[key][code][_]['columns'][:]
                    columns = ['{}@{}'.format(_, i.decode('utf-8')) for i in columns]
                    values = files[key][code][_]['values'][:]

                    tmp_df = pd.DataFrame(values)
                    tmp_df.columns = columns
                    tmp_list.append(tmp_df)

            df = pd.concat(tmp_list, axis=1)
            df['code'] = code
            df_list.append(df)

        df_daily = pd.concat(df_list, axis=0, sort=True)
        df_daily['date'] = key
        df_daily_list.append(df_daily)

    df_train = pd.concat(df_daily_list, axis=0, sort=True)
    assert 1 == 2
