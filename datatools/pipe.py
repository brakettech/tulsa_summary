from daq.pico import CSV
from dask import compute, delayed
from .harmonic import Harmonic
from scipy import signal
import dask.multiprocessing
import numpy as np
import os
import pandas as pd
import textwrap

class Pipe:
    def __init__(self, df_log_batch, channel_mapper, n_jobs=1, verbose=True, harmonic=3):
        """
        Initialize processor with a dataframe containing the batch
        of files that belong to one pipe.

        The Dataframe must always contain the columns "pos" and "file_name".
        All other columns are ignored.

        pipe = any python object that identifies the pipe
        data_dir = path_to_csv_data_files (defaults to "./")
        n_jobs = the number of jobs to run in parallel
        """
        self.channel_mapper = channel_mapper
        self.n_jobs = n_jobs
        self.harmonics = [1, harmonic]
        self.verbose = verbose

        self.df_log = df_log_batch
        self._df = None

    @property
    def df(self):
        return self._df.copy()

    @property
    def df_indexed(self):
        return self._indexify(self._df)

    @property
    def column_descriptions(self):
        return textwrap.dedent(
            """
            Column descriptions for result dataframe.

            'key1': first key field
            'key2': second key field
            "key*': * key field
            "....': ....
            'pos': the position of a given measurement
            'prim_sec_amp': amplitude of V_secondary / I_primary
            'prim_sec_phi': phase of V_secondary / I_primary
            'prim_rec_amp': amplitude of V_receiver / I_primary
            'prim_rec_phi: phase of V_receiver / I_primary
            'sec_rec_amp': amplitude of V_receiver / V_secondary
            'sec_rec_phi': phase of V_receiver / V_secondary
            'sec_harm_db': third harmonic power fraction of secondary in dB
            'rec_harm_db': third harmonic power fraction of receiver in dB
            """
        )

    def filter(self, df):
        filter_cols = ['res_volt', 'sec_volt', 'rec_volt']
        for col in filter_cols:
            # 8 pol filter at .01 of nyquist
            b, a = signal.butter(8, 0.01)
            df.loc[:, col] = signal.filtfilt(b, a, df[col].values, padlen=150)
        return df

    def _process_file(self, file_name, key_dict):
        if self.verbose:
            print(f'Processing {file_name}')

        df = CSV(file_name=file_name, max_sample_freq=1e9, **self.channel_mapper).df
        df = self.filter(df)

        # In everything below the h_ prefix stands for "harmonic" because the
        # variable contains an instance of a harmonic object.

        # fit the primary current
        h_i_prim = Harmonic(harmonics=self.harmonics)
        h_i_prim.fit(df.t, df.res_volt)

        # Induced voltage proportional to derivative of current
        h_i_prim = h_i_prim.derivative()

        # fit the secondary voltage
        h_v_sec = Harmonic(harmonics=self.harmonics)
        h_v_sec.fit(df.t, df.sec_volt)

        # fit the receiver voltage
        h_v_rec = Harmonic(harmonics=self.harmonics)
        h_v_rec.fit(df.t, df.rec_volt)

        # compute "impedence" objects
        h_z_prim_sec = h_v_sec / h_i_prim
        h_z_prim_rec = h_v_rec / h_i_prim
        h_z_sec_rec = h_v_rec / h_v_sec

        out = key_dict

        # populate some results
        rec = dict(
            # the "impedence" of secondary with respect to primary
            prim_sec_amp=h_z_prim_sec.amplitudes[0],
            prim_sec_phi=h_z_prim_sec.phases[0],

            # the "impedence" of receiver with respect to primary
            prim_rec_amp=h_z_prim_rec.amplitudes[0],
            prim_rec_phi=h_z_prim_rec.phases[0],

            # the "impedence" of receiver with respect to secondary
            sec_rec_amp=h_z_sec_rec.amplitudes[0],
            sec_rec_phi=h_z_sec_rec.phases[0],

            # the relative power in the 3rd harmonic expressed in db
            sec_harm_db=10 * np.log10(h_v_sec.amplitudes[1] ** 2 / np.sum(h_v_sec.amplitudes ** 2)),
            rec_harm_db=10 * np.log10(h_v_rec.amplitudes[1] ** 2 / np.sum(h_v_rec.amplitudes ** 2)),
        )
        out.update(rec)
        return out

    def process(self, indexed=False):
        """
        Process all files belonging to a single pipe.

        indexed=True will create multi-indexes for rows and columns
        """

        def fake_delayed(func):
            """
            This behaves as a mock for dask.delayed when you are debugging
            and don't want parallel execution.  It just returns the "delayed"
            function without applying any delay.
            """
            return func

        def fake_compute(*future_list, **kwargs):
            """
            This bahaves as a mock for dask.compute when you are debugging and
            don't want parallel execution.  In this case, the "futures" are just
            the actual results returned by the non-delayed functions wrapped by
            the fake_delay function.
            """
            return future_list

        # if instructed to run in parallel, use dask.delayed and dask.compute
        # otherwise just use the mocks
        if self.n_jobs > 1:
            delay_func = delayed
            compute_func = compute
        else:
            delay_func = fake_delayed
            compute_func = fake_compute

        # initialize a list to hold futures
        future_list = []

        key_names = [c for c in self.df_log.columns if c != 'file_name']
        for _, row in self.df_log.iterrows():
            file_name = row['file_name']
            key_dict = {k: row[k] for k in key_names}

            # create a future (this directly evalutes the function if n_jobs < 2)
            future = delay_func(self._process_file)(file_name, key_dict)
            future_list.append(future)

        # evaluate all futures (if n_jobs < 2), this just puts results into the rec_list
        rec_list = list(compute_func(*future_list, scheduler='processes'))

        # specify proper column ordering for output
        columns = list(key_dict.keys()) + [
            'prim_sec_amp',
            'prim_sec_phi',
            'prim_rec_amp',
            'prim_rec_phi',
            'sec_rec_amp',
            'sec_rec_phi',
            'sec_harm_db',
            'rec_harm_db',
        ]

        # create and return a results frame
        df = pd.DataFrame(rec_list, columns=columns)
        self._df = df

        return self.df

    def _indexify(self, df):
        arrays = [
            ['primary', 'primary', 'primary', 'primary', 'sec', 'sec', 'primary', 'primary'],
            ['secondary', 'secondary', 'receiver', 'receiver', 'receiver', 'receiver', 'secondary', 'secondary'],
            ['amplitude', 'phase', 'amplitude', 'phase', 'amplitude', 'phase', 'db', 'db']
        ]

        df = df.set_index(['pipe', 'pos'])
        df.columns = pd.MultiIndex.from_arrays(arrays, names=['driver', 'detector', 'quantity'])
        df = df.sort_index(axis=1).sort_index()
        return df
