#! /usr/bin/env python

import os
import re

import numpy as np
import pandas as pd

from datatools.pipe import Pipe
import easier as ezr
from daq.pico import CSV


class Procesor:
    # this is the maximum number of identical samples to take
    max_samples = 2

    # Mapper holding channel names for data files
    channel_mapper = dict(
        a='sig_gen',
        b='res_volt',
        c='rec_volt',
        d='sec_volt'
    )

    all_files_log = './test_logs/all_runs.csv'
    files_to_analyze_log = './test_logs/samples_to_analyze.csv'
    data_dir = './data_dir'

    results_file = 'results.csv'

    @classmethod
    def _get_log_file(cls):
        # Get the CSV of the files that should be analyzed
        dfa = pd.read_csv('./test_logs/samples_to_analyze.csv')

        # Get CSV file of all files created, and limit to useful columns
        dfl = pd.read_csv('./test_logs/all_runs.csv')
        columns = [
            'test_no',
            'sample',
            'bends',
            'primary_position',
            'frequency',
            'temperature',
            'file_name',
        ]
        dfl = dfl[columns]

        # Join into master log frame and return
        dfj = pd.merge(dfl, dfa, on=['sample'], how='inner').reset_index(drop=True)
        dfj = dfj.rename(columns={'file_name': 'file_tag'})
        return dfj

    @classmethod
    def _find_files(cls, data_dir, max_samples):
        """
        Find all data files under a specified directory
        """
        if not os.path.isdir(data_dir):
            raise ValueError(f'Path {data_dir} does not exist')
        # the regex pattern for identifying a data file
        rex_file = re.compile(r'.*/?\d+\-\d+(_\d+)?\.csv')

        # initialize empty list of data files
        data_files = []

        # recursively search data directory
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                # only consider data files
                if rex_file.match(file):

                    # compute the full path to the datafile
                    file_name = os.path.join(root, file)

                    # the file_tag is what is put into the log file
                    file_tag = re.sub(r'(_\d+)?.csv', '', file)

                    # add the data file
                    data_files.append((file_tag, file_name))

        # create and return the output dataframe
        dff = pd.DataFrame(data_files, columns=['file_tag', 'file_name'])

        def compute_sample_num(batch):
            """
            For each condition measured, this function computes the sample numbe
            """
            batch.insert(1, 'sample_num', np.array(range(len(batch))) + 1)
            return batch

        # These fields identify measurement conditions (that can be sample multiple times)
        grouping_fields = ['file_tag']

        # This will order by filename within batches, although it's no clear that's needed
        sorting_fields = grouping_fields + ['file_name']
        dff = dff.sort_values(by=sorting_fields)

        # Actually run the groupby to assign sample number
        dff = dff.groupby(by=grouping_fields).apply(compute_sample_num)

        # Limit file frame to the max number of samples per file tag
        dff = dff[dff.sample_num <= max_samples]

        # Get rid of unneeded field
        dff = dff.drop('sample_num', axis=1)

        return dff

    @classmethod
    def _get_processing_frame(cls, data_dir, max_samples):
        # Get the log file
        dfl = cls._get_log_file()

        # Get the data files
        dff = cls._find_files(data_dir, max_samples)

        # Create the processing frame and return it
        dfj = pd.merge(dff, dfl, on='file_tag')
        return dfj

    def run(self, n_jobs=2):
        with ezr.Timer('processing'):
            dfp = self._get_processing_frame(self.data_dir, self.max_samples)
            pipe = Pipe(dfp, channel_mapper=self.channel_mapper, n_jobs=n_jobs)
            dfr = pipe.process()
            dfr.to_csv(self.results_file, index=False)


if __name__ == '__main__':
    proc = Procesor()
    proc.run(n_jobs=2)


