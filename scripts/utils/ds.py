import gc
import os

import numpy as np
import xarray as xr
import xgboost as xgb
from torch.utils.data import Dataset
import json


class TD(Dataset):
    """
    abstract class
    """
    def __init__(self) -> None:
        super(TD, self).__init__()

    def normlize(self, data:np.ndarray, name:str):
        # handle nan
        data_ = data[~np.isnan(data)]
        if self.flag == "train":
            self.dist[name] = {
                "mean": float(data_.mean()),
                "std": float(data_.std())
            }
        mean = self.dist[name]["mean"]
        std = self.dist[name]["std"]
        data = np.nan_to_num(data, nan=mean)
        return (data - mean) / std

    def _handle_edh(self, _edh:list):
        edh = [None] * len(_edh)
        time = [None] * len(_edh)
        for i in range(len(_edh)):
            edh[i] = _edh[i].data
            time[i] = _edh[i].time.data
        edh = np.concatenate(edh, axis=0)
        time = np.concatenate(time, axis=0)
        return (edh, time)

    def _handle_u10(self, _era5:list):
        u10 = [None] * len(_era5)
        for i in range(len(_era5)):
            u10[i] = _era5[i].u10.data
        u10 = np.concatenate(u10, axis=0)
        # delete NaN and Normalize
        u10 = self.normlize(u10, "u10")
        return u10

    def _handle_v10(self, _era5:list):
        v10 = [None] * len(_era5)
        for i in range(len(_era5)):
            v10[i] = _era5[i].v10.data
        v10 = np.concatenate(v10, axis=0)
        # delete NaN and Normalize
        v10 = self.normlize(v10, "v10")
        return v10

    def _handle_t2m(self, _era5:list):
        t2m = [None] * len(_era5)
        for i in range(len(_era5)):
            t2m[i] = _era5[i].t2m.data
        t2m = np.concatenate(t2m, axis=0)
        # delete NaN and Normalize
        t2m = self.normlize(t2m, "t2m")
        return t2m

    def _handle_msl(self, _era5:list):
        msl = [None] * len(_era5)
        for i in range(len(_era5)):
            msl[i] = _era5[i].msl.data
        msl = np.concatenate(msl, axis=0)
        # delete NaN and Normalize
        msl = self.normlize(msl, "msl")
        return msl

    def _handle_sst(self, _era5:list):
        sst = [None] * len(_era5)
        for i in range(len(_era5)):
            sst[i] = _era5[i].sst.data
        sst = np.concatenate(sst, axis=0)
        # delete NaN and Normalize
        sst = self.normlize(sst, "sst")
        return sst

    def _handle_q2m(self, _era5:list):
        q2m = [None] * len(_era5)
        for i in range(len(_era5)):
            q2m[i] = _era5[i].q2m.data
        q2m = np.concatenate(q2m, axis=0)
        # delete NaN and Normalize
        q2m = self.normlize(q2m, "q2m")
        return q2m

class ERA5Dataset(TD):

    def __init__(self, edh, era5, flag="train") -> None:
        super(ERA5Dataset, self).__init__()
        self.flag = flag
        if flag == "train":
            self.dist = {}
        else:
            # load mean and std
            with open('.cache/dist.json', 'r') as f:  
                self.dist = json.load(f) 
        # load edh(time, lat, lon)
                
        _edh = []
        for root, _, files in os.walk(edh):  
            for filename in files:
                _edh.append(
                    xr.load_dataset(os.path.join(root, filename)).EDH
                )
        _edh = sorted(
            _edh, key=lambda _edh: _edh.time.data[0]
        )
        self.lon = _edh[0].longitude.data
        self.lat = _edh[0].latitude.data
        self.edh, self.time = self._handle_edh(_edh)
        gc.collect()

        # load era5=[time, lat, lon]
        _era5 = []
        for root, _, files in os.walk(era5):  
            for filename in files:
                _era5.append(
                    xr.load_dataset(os.path.join(root, filename))
                )
        _era5 = sorted(
            _era5, key=lambda _era5: _era5.time.data[0]
        )
        self.u10 = self._handle_u10(_era5)
        gc.collect()
        self.v10 = self._handle_v10(_era5)
        gc.collect()
        self.t2m = self._handle_t2m(_era5)
        gc.collect()
        self.msl = self._handle_msl(_era5)
        gc.collect()
        self.sst = self._handle_sst(_era5)
        gc.collect()
        self.q2m = self._handle_q2m(_era5)
        gc.collect()
        if flag == "train":
            with open('.cache/dist.json', 'w') as f:  
                json.dump(self.dist, f)

    def __len__(self) -> int:
        return len(self.time)

    def __getitem__(self, index: int):
        return (
            np.array(
                [
                    self.u10[index], 
                    self.v10[index], 
                    self.t2m[index],
                    self.msl[index], 
                    self.sst[index], 
                    self.q2m[index]
                ]
            ),
            np.expand_dims(self.edh[index], axis=0)
        )

class ERA5Iterator(xgb.DataIter):
    def __init__(
            self, 
            data,
            labels,
            batch_size,
    ) -> None:
        self._data = data
        self._label = labels
        self._batch_size = batch_size
        self._it = 0
        self._num_samples = self._data.shape[0]
        self._num_batches = np.ceil(self._num_samples / batch_size)
        super().__init__(cache_prefix=".cache/")

    def next(self, input_data):
        if self._it == self._num_batches:
            return 0
        else:
            start = self._it * self._batch_size
            end = min((self._it + 1) * self._batch_size, self._num_samples)
            input_data(data=self._data[start:end], label=self._label[start:end])
            self._it += 1
            return 1

    def reset(self):
        self._it = 0