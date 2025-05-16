from pathlib import Path
from typing import Union, List
import numpy as np
import pandas as pd
from datasets import Dataset

def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    start_time: str = "2024-01-05 00:00",
    time_step: str = "1H",
):
    assert isinstance(time_series, list) or (
        isinstance(time_series, np.ndarray) and time_series.ndim == 2
    )

    start = pd.Timestamp(start_time)
    time_step_delta = pd.Timedelta(time_step)

    records = []
    for ts in time_series:
        timestamps = [start + i * time_step_delta for i in range(len(ts))]
        records.append({
            "start": str(start),
            "timestamps": [str(t) for t in timestamps],
            "target": ts.tolist()
        })

    dataset = Dataset.from_list(records)
    dataset.save_to_disk(path)

if __name__ == "__main__":
    DomainDataset = {
        'Energy': ['CAISO_AL', 'NYISO_AL', 'Wind'],
        'Healthcare': ['Covid-19', 'illness'],
        'Environment': ['weather', 'AQShunyi', 'AQWan'],
        'Nature': ['CzeLan', 'ZafNoo'],
        'Traffic': ['M_DENSE', 'METR_LA', 'NYCTAXI_Inflow', 'NYCTAXI_Outflow'],
        'Finance': ['Exchange_rate', 'IBM_m', 'MSFT_w', 'NASDAQ_w'],
        'Web': ['Wike2000', 'BizITObs-Application', 'BizITObs-Service', 'BizITObs-L2C'],
        'CloudOps': ['Bitbrains-rnd100-cpuusage', 'Bitbrains-rnd100-memoryusage']
    }

    DomainFreq = {
        'Energy': ['1H', '1H', '15min'],
        'Healthcare': ['1D', '7D'],
        'Environment': ['10min', '1H', '1H'],
        'Nature': ['30min', '30min'],
        'Traffic': ['1H', '5min', '1H', '1H'],
        'Finance': ['1D', '30D', '7D', '7D'],
        'Web': ['1D', '10S', '10S', '5min'],
        'CloudOps': ['15min', '15min']
    }

    Start_time = {
        'Energy': ['2024-01-05 00:00', '2023-01-06 00:00', '2020-01-01 00:00'],
        'Healthcare': ['2020-01-03 00:00', '2002-01-01 00:00'],
        'Environment': ['2020-01-01 00:10', '2013-03-01 00:00', '2013-03-01 00:00'],
        'Nature': ['2016-05-06 00:11', '2008-05-15 23:20'],
        'Traffic': ['2018-01-01 00:00', '2012-03-01 00:00', '2020-01-01 00:00', '2020-01-01 00:00'],
        'Finance': ['1990-01-01 00:00', '1962-01-01 00:00', '1986-03-10 00:00', '2010-01-04 00:00'],
        'Web': ['2012-01-01 00:00', '2023-03-13 10:01:50', '2023-03-13 10:01:50', '2020-11-01 00:00'],
        'CloudOps': ['2013-08-12 13:30', '2013-08-12 13:30']
    }

    for domain in ['Energy', 'Healthcare', 'Environment', 'Traffic', 'Finance', 'Web_CloudOps']:
        for i in range(len(DomainDataset[domain])):
            dataset = DomainDataset[domain][i]
            freq = DomainFreq[domain][i]
            start_time = Start_time[domain][i]

            data_read = pd.read_csv(f'/home/xiongshiwei/chronos-forecasting/Dataset/{domain}/{dataset}.csv')
            time_series = data_read.iloc[:, 1:].values.T  
            train = data_read.iloc[:int(data_read.shape[0] * 0.7), 1:].values.T 

            train_mean = np.mean(train, axis=1, keepdims=True)
            train_std = np.std(train, axis=1, keepdims=True)
            train_std = np.where(train_std == 0, 1, train_std)

            time_series_normalized = (time_series - train_mean) / train_std
            train_normalized = (train - train_mean) / train_std

            convert_to_arrow(
                f"/home/xiongshiwei/chronos-forecasting/data/{dataset}",
                time_series=time_series_normalized,
                start_time=start_time,
                time_step=freq
            )

            convert_to_arrow(
                f"/home/xiongshiwei/chronos-forecasting/data/Train/{dataset}",
                time_series=train_normalized,
                start_time=start_time,
                time_step=freq
            )












































# def convert_to_arrow(
#     path: Union[str, Path],
#     time_series: Union[List[np.ndarray], np.ndarray],
#     compression: str = "lz4",
# ):
#     """
#     Store a given set of series into Arrow format at the specified path.

#     Input data can be either a list of 1D numpy arrays, or a single 2D
#     numpy array of shape (num_series, time_length).
#     """
#     assert isinstance(time_series, list) or (
#         isinstance(time_series, np.ndarray) and
#         time_series.ndim == 2
#     )

#     # Set an arbitrary start time
#     # start = np.datetime64("2000-01-01 00:00", "s")
#     start = np.datetime64("2024-01-05 00:00", "s")  # 重写起始时间点

#     dataset = [
#         {"start": start, "target": ts} for ts in time_series
#     ]

#     # dataset = []
#     # for i, ts in enumerate(time_series):
#     #     # 计算当前时间点
#     #     current_start = start + np.timedelta64(i, "h")
#     #     # 添加到数据集
#     #     dataset.append({"start": current_start, "target": ts})
    
#     print(dataset)

#     ArrowWriter(compression=compression).write_to_file(
#         dataset,
#         path=path,
#     )


# if __name__ == "__main__":
#     # Generate 20 random time series of length 1024
#     # time_series = [np.random.randn(1024) for i in range(20)]

#     data_read = pd.read_csv('/home/xiongshiwei/chronos-forecasting/Dataset/Energy/CAISO_AL.csv')
#     time_series = data_read.iloc[:,1:].values.T

#     print(time_series)

#     # Convert to GluonTS arrow format
#     # convert_to_arrow("./dataset/arrow/noise-data.arrow", time_series=time_series)
#     convert_to_arrow("/home/xiongshiwei/chronos-forecasting/data/CAISO_AL.arrow", time_series=time_series)





# from pathlib import Path
# from typing import List, Union

# import numpy as np
# from gluonts.dataset.arrow import ArrowWriter


# def convert_to_arrow(
#     path: Union[str, Path],
#     time_series: Union[List[np.ndarray], np.ndarray],
#     compression: str = "lz4",
# ):
#     """
#     Store a given set of series into Arrow format at the specified path.

#     Input data can be either a list of 1D numpy arrays, or a single 2D
#     numpy array of shape (num_series, time_length).
#     """
#     assert isinstance(time_series, list) or (
#         isinstance(time_series, np.ndarray) and
#         time_series.ndim == 2
#     )

#     # Set an arbitrary start time
#     start = np.datetime64("2000-01-01 00:00", "s")

#     dataset = [
#         {"start": start, "target": ts} for ts in time_series
#     ]

#     ArrowWriter(compression=compression).write_to_file(
#         dataset,
#         path=path,
#     )


# if __name__ == "__main__":
#     # Generate 20 random time series of length 1024
#     time_series = [np.random.randn(1024) for i in range(5)]

#     # Convert to GluonTS arrow format
#     convert_to_arrow("./noise-data.arrow", time_series=time_series)