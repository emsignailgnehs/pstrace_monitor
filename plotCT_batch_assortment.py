#%% imports
from utils._util import ViewerDataSource
import matplotlib.pyplot as plt
import numpy as np
from utils.calling_algorithm import *
from sklearn.pipeline import Pipeline
import textwrap
import csv
from itertools import combinations
import time
import tkinter as tk
from tkinter import filedialog
import glob, os
import json
import re
from pathlib import Path
import sys

"""
<<Instruction>>
To use this script to convert batch file, the files needs to be organized
in the following folder structure:

{root}\\yyyy-mm\\yyyymmdd\\filename

while {root} refers to the path of the root directory
"""

################################################################################
#### pickle file to plot data from                                          ####
#### """                                                                    ####
#### ██████╗ ██╗ ██████╗██╗  ██╗██╗     ███████╗███████╗██╗██╗     ███████╗ ####
#### ██╔══██╗██║██╔════╝██║ ██╔╝██║     ██╔════╝██╔════╝██║██║     ██╔════╝ ####
#### ██████╔╝██║██║     █████╔╝ ██║     █████╗  █████╗  ██║██║     █████╗   ####
#### ██╔═══╝ ██║██║     ██╔═██╗ ██║     ██╔══╝  ██╔══╝  ██║██║     ██╔══╝   ####
#### ██║     ██║╚██████╗██║  ██╗███████╗███████╗██║     ██║███████╗███████╗ ####
#### ╚═╝     ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝     ╚═╝╚══════╝╚══════╝ ####
#### Change this manually if running code in terminal.                      ####
################################################################################
# picklefile = r"C:\Users\hui\RnD\Projects\LAMP-Covid Sensor\Data Export\20220223\20220223 SL fresh vs store caps_ fresh vs stored sensors.picklez"
#%% user end input

class IO_Paths:
    def __init__(self, file_suffix= '.picklez'):
        self.get_user_inputs()
        self.get_filepaths(file_suffix)

    def get_user_inputs(self):
        self.root_folder = self.ui_getdirectory()
        savename = input("Please input the name of the output batch file...\n")
        self.picklefile = Path(fr'{self.root_folder}\\{savename}.csv')

    def get_filepaths(self, file_suffix):
        primaryfolders = [dir[1] for dir in os.walk(self.root_folder)][0]
        primarypaths = [Path(fr'{self.root_folder}\{folder}') for folder in primaryfolders if re.findall(r'\d{4}[-]\d{2}', folder)]
        secondaryfolder_paths = []
        for primarypath in primarypaths:
            secondaryfolders = [dir[1] for dir in os.walk(primarypath)][0]
            secondaryfolder_paths += [Path(fr'{primarypath}\\{secondaryfolder}') for secondaryfolder in secondaryfolders]
        filepaths = []
        for secondaryfolder_path in secondaryfolder_paths:
            filenames = [dir[2] for dir in os.walk(secondaryfolder_path)][0]
            filepaths += [Path(fr'{secondaryfolder_path}\\{filename}') for filename in filenames if filename.endswith(file_suffix)]
        self.filepaths = filepaths
    
    def ui_getdirectory(self):
        root = tk.Tk() # pointing root to Tk() to use it as Tk() in program.
        root.withdraw() # Hides small tkinter window.
        root.attributes('-topmost', True) # Opened windows will be active. above all windows despite of selection.
        return filedialog.askdirectory(title= 'Please select the root directory')

class PlotCT(IO_Paths):
    def __init__(self):
        IO_Paths.__init__(self)
        self.run_analysis(self.filepaths)
    
    def load_pickleFile(self, filepath: Path):
        dataSource = ViewerDataSource()
        pickleFiles = [filepath]
        dataSource.load_picklefiles(pickleFiles)
        data = dataSource.rawView.get('data',[])
        results = []
        for d in data:
            t = timeseries_to_axis(d['data']['time'])
            pc = [i['pc'] for i in d['data']['fit']]
            results.append([(t,pc), True, d.get('name','No Name'),d.get('_channel','Unknown'),d.get('name','C0')[-2:]]) # '_channel' is actually the device name.
        results.sort(key=lambda x:(x[4],x[2]))
        traces = [i[0] for i in results]
        userMark = [i[1] for i in results]
        names = [i[2] for i in results]
        devices = [i[3] for i in results]
        dataset = convert_list_to_X(traces),np.array(userMark),np.array(names),np.array(devices)
        rearr_dataset = removeDuplicates(*dataset)
        return rearr_dataset
    
    def sort_pickleFile(self, rearr_dataset):
        X, y, names,devices = rearr_dataset
        sorted_names = self.sort_datasets(names)
        idx_flag = [list(names).index(name) for name in sorted_names]

        X = self.sort_by_idxflag(X, idx_flag)
        names = self.sort_by_idxflag(names, idx_flag)
        devices = self.sort_by_idxflag(devices, idx_flag)
        y = np.array([1] * len(X))
        sorted_dataset = X, y, names,devices
        return sorted_dataset

    def calc_param(self, sorted_dataset):
        X, y, names,devices = sorted_dataset
        cutoffStart = 5
        cutoffEnd = 30
        normStart = 5
        normEnd = 10

        hCtT = Pipeline([('smooth', Smoother(stddev=2, windowlength=11, window='hanning')),
                         ('normalize', Normalize(mode='mean', normalizeRange=(normStart, normEnd))),
                         ('truncate', Truncate(cutoffStart=cutoffStart, cutoffEnd=cutoffEnd, n=90)),
                         ('Derivitive', Derivitive(window=31, deg=3)),
                         ('peak', FindPeak()),
                         ('logCt',HyperCt()),])
        hCtT_X = hCtT.transform(X)

        hCtTPredictT = Pipeline([('smooth', Smoother(stddev=2, windowlength=11, window='hanning')),
                                 ('normalize', Normalize(mode='mean', normalizeRange=(normStart, normEnd))),
                                 ('truncate', Truncate(cutoffStart=cutoffStart, cutoffEnd=cutoffEnd, n=90)),
                                 ('Derivitive', Derivitive(window=31, deg=3)),
                                 ('peak', FindPeak()),
                                 ('logCt',HyperCt()),
                                 ('predictor',SdPrPredictor(prominence=0.2,sd=0.106382))])
        hCtpred_X = hCtTPredictT.transform(X)
        
        calcd_dataset = hCtT_X, hCtpred_X, y, names, devices
        return calcd_dataset

    def run_analysis(self, filepaths: list):
        with open(self.picklefile, 'w') as f:
            f.write('"id","sample type","reader id","date","c1 result","c1 ct","c1 pr","c1 sd","c4 result","c4 ct","c4 pr","c4 sd","led result", "originalPickleLocation"\n')
        for i, filepath in enumerate(filepaths):
            filepath = str(filepath)
            print(f'Current progress: {i+1}/{len(filepaths)}')
            rearr_dataset = self.load_pickleFile(filepath)
            sorted_dataset = self.sort_pickleFile(rearr_dataset) #sorting furnction for clinical data
            calcd_dataset = self.calc_param(sorted_dataset)
            linedataset = self.data_to_linedata(calcd_dataset)
            for linedata in linedataset:
                self.write_output(linedata, filepath)

    def data_to_linedata(self, calcd_dataset):
        # generator to yield each row of data as a dictionary
        for i in range(len(calcd_dataset[0])):
            output = {'hCtT_X': None,'hCtpred_X': None,'y': None, 'name': None, 'device': None}
            for key, val_array in zip(output, calcd_dataset):
                output[key] = val_array[i]
            yield output
    
    def write_output(self, linedata, filepath, result = ['']):
        id = ''.join(linedata['name'].split('-')[:-1])
        ch = linedata['name'].split('-')[-1]
        reader_id = linedata['device'].split('-')[-1]
        ct, pr, sd = [linedata['hCtT_X'][-1], linedata['hCtT_X'][1], linedata['hCtT_X'][5]]
        date = filepath.split('\\')[-2]
        calling =  'Positive' if linedata['hCtpred_X'][0] else 'Negative'
        if ch == 'C1':
            result[0] += '+' if calling == 'Positive' else '-'
            msg = f'{id},Saliva,{reader_id},{date},{calling},{ct},{pr},{sd},'
        elif ch == 'C4':
            result[0] += '+' if calling == 'Positive' else '-'
            if result[0] == '-+':
                led_result = 'Negative'
            elif result[0] == '--':
                led_result = 'Invalid'
            else:
                led_result = 'Positive'
            msg = f'{calling}, {ct}, {pr}, {sd}, {led_result}, {filepath}\n'
            result[0] = ''
        with open(self.picklefile, 'a') as f:
            f.write(msg)
    
    def sort_by_idxflag(self, array, idx_flag):
        return np.array([array[i] for i in idx_flag])

    def sort_datasets(self, datasets):
        #generic sorting based on the names of the measurements
        #generalized naming: NAME - C1
        #                     |     |
        #                    id    ch
        def sorter_id(string_to_sort: str) -> str:
            return ''.join(string_to_sort.split('-')[:-1])
        def sorter_ch(string_to_sort: str) -> str:
            return int(string_to_sort.split('-')[-1][-1])
        return sorted(datasets, key = lambda x: (sorter_id(x), sorter_ch(x)))

class PlotCT_Clinical(PlotCT):
    def __init__(self, file_suffix= '_Clinical.picklez'):
        IO_Paths.__init__(self, file_suffix)
        self.run_analysis(self.filepaths)
    
    # def load_pickleFile(self, filepath: Path):
    #     dataSource = ViewerDataSource()
    #     pickleFiles = [filepath]
    #     dataSource.load_picklefiles(pickleFiles)
    #     data = dataSource.rawView.get('data',[])
    #     results = []
    #     for d in data:
    #         t = timeseries_to_axis(d['data']['time'])
    #         pc = [i['pc'] for i in d['data']['fit']]
    #         results.append([(t,pc), True, d.get('name','No Name'),d.get('_channel','Unknown'),d.get('name','C0')[-2:]]) # '_channel' is actually the device name.
    #     results.sort(key=lambda x:(x[4],x[2]))
    #     traces = [i[0] for i in results]
    #     userMark = [i[1] for i in results]
    #     names = [i[2] for i in results]
    #     devices = [i[3] for i in results]
    #     dataset = convert_list_to_X(traces),np.array(userMark),np.array(names),np.array(devices)
    #     rearr_dataset = removeDuplicates(*dataset)
    #     return rearr_dataset
    
    # def sort_pickleFile(self, rearr_dataset):
    #     X, y, names,devices = rearr_dataset
    #     sorted_names = self.sort_datasets(names)
    #     idx_flag = [list(names).index(name) for name in sorted_names]

    #     X = self.sort_by_idxflag(X, idx_flag)
    #     names = self.sort_by_idxflag(names, idx_flag)
    #     devices = self.sort_by_idxflag(devices, idx_flag)
    #     y = np.array([1] * len(X))
    #     sorted_dataset = X, y, names,devices
    #     return sorted_dataset

    # def calc_param(self, sorted_dataset):
    #     X, y, names,devices = sorted_dataset
    #     cutoffStart = 5
    #     cutoffEnd = 30
    #     normStart = 5
    #     normEnd = 10

    #     hCtT = Pipeline([('smooth', Smoother(stddev=2, windowlength=11, window='hanning')),
    #                      ('normalize', Normalize(mode='mean', normalizeRange=(normStart, normEnd))),
    #                      ('truncate', Truncate(cutoffStart=cutoffStart, cutoffEnd=cutoffEnd, n=90)),
    #                      ('Derivitive', Derivitive(window=31, deg=3)),
    #                      ('peak', FindPeak()),
    #                      ('logCt',HyperCt()),])
    #     hCtT_X = hCtT.transform(X)

    #     hCtTPredictT = Pipeline([('smooth', Smoother(stddev=2, windowlength=11, window='hanning')),
    #                              ('normalize', Normalize(mode='mean', normalizeRange=(normStart, normEnd))),
    #                              ('truncate', Truncate(cutoffStart=cutoffStart, cutoffEnd=cutoffEnd, n=90)),
    #                              ('Derivitive', Derivitive(window=31, deg=3)),
    #                              ('peak', FindPeak()),
    #                              ('logCt',HyperCt()),
    #                              ('predictor',SdPrPredictor(prominence=0.2,sd=0.106382))])
    #     hCtpred_X = hCtTPredictT.transform(X)
        
    #     calcd_dataset = hCtT_X, hCtpred_X, y, names, devices
    #     return calcd_dataset

    # def run_analysis(self, filepaths: list):
    #     with open(self.picklefile, 'w') as f:
    #         f.write('"id","sample type","reader id","date","c1 result","c1 ct","c1 pr","c1 sd","c4 result","c4 ct","c4 pr","c4 sd","led result", "originalPickleLocation"\n')
    #     for i, filepath in enumerate(filepaths):
    #         filepath = str(filepath)
    #         print(f'Current progress: {i+1}/{len(filepaths)}')
    #         rearr_dataset = self.load_pickleFile(filepath)
    #         sorted_dataset = self.sort_pickleFile(rearr_dataset) #sorting furnction for clinical data
    #         calcd_dataset = self.calc_param(sorted_dataset)
    #         linedataset = self.data_to_linedata(calcd_dataset)
    #         for linedata in linedataset:
    #             self.write_output(linedata, filepath)

    # def data_to_linedata(self, calcd_dataset):
    #     # generator to yield each row of data as a dictionary
    #     for i in range(len(calcd_dataset[0])):
    #         output = {'hCtT_X': None,'hCtpred_X': None,'y': None, 'name': None, 'device': None}
    #         for key, val_array in zip(output, calcd_dataset):
    #             output[key] = val_array[i]
    #         yield output
    
    def write_output(self, linedata, filepath, result = ['']):
        sample_type, id, ch = linedata['name'].split('-')
        reader_id = linedata['device'].split('-')[-1]
        ct, pr, sd = [linedata['hCtT_X'][-1], linedata['hCtT_X'][1], linedata['hCtT_X'][5]]
        date = filepath.split('\\')[-2]
        calling =  'Positive' if linedata['hCtpred_X'][0] else 'Negative'
        if ch == 'C1':
            result[0] += '+' if calling == 'Positive' else '-'
            msg = f'{id}, {sample_type}, {reader_id}, {date}, {calling}, {ct}, {pr}, {sd},'
        elif ch == 'C4':
            result[0] += '+' if calling == 'Positive' else '-'
            if result[0] == '-+':
                led_result = 'Negative'
            elif result[0] == '--':
                led_result = 'Invalid'
            else:
                led_result = 'Positive'
            msg = f'{calling}, {ct}, {pr}, {sd}, {led_result}, {filepath}\n'
            result[0] = ''
        with open(self.picklefile, 'a') as f:
            f.write(msg)
    
    # def sort_by_idxflag(self, array, idx_flag):
    #     return np.array([array[i] for i in idx_flag])

    def sort_datasets(self, datasets):
        #specific sorting function for clinical trial naming system
        #generalized naming: H NAS - 0163 - C1
        #                    |  |      |     |
        #                  cat type   id    ch
        def sorter_idnum(array):
            return array.split('-')[1]
        def sorter_type(array):
            return array.split('-')[0][1:]
        def sorter_cat(array):
            return array.split('-')[0][0]
        def sorter_ch(array):
            return int(array.split('-')[-1][-1])
        return sorted(datasets, key = lambda x: (sorter_idnum(x), sorter_type(x), sorter_cat(x), sorter_ch(x)))

if __name__ == '__main__':
    try:
        test_type = sys.argv[1]
        if test_type in ['-c', 'clinical']:
            print('Running Clinical Compiler')
            PlotCT_Clinical()
        else:
            print('Running Regular Compiler')
            PlotCT()
    except:
        print('No Test Type Assigned... Running Regular Compiler')
        PlotCT()
    #
    # first_row = True
    # for folder in folders:
    #     subfolder = f'{root_folder}\{folder}'
    #     files = [dir[2] for dir in os.walk(subfolder)][0]
    #     files = [file for file in files if re.findall(r'.*(_clinical.*\.picklez)$', file.lower())]
    #     for file in files:
    #         filepath = f'{subfolder}\{file}'
    #         #%% load data
    #         print(f'File you entered is: {file}')
    #         print('reading data...')
    #         dataSource = ViewerDataSource()
    #         pickleFiles = [filepath]
    #         dataSource.load_picklefiles(pickleFiles)
    #
    #         data = dataSource.rawView.get('data',[])
    #         results = []
    #         for d in data:
    #             t = timeseries_to_axis(d['data']['time'])
    #             pc = [i['pc'] for i in d['data']['fit']]
    #             # this '_channel' is actually the device name.
    #             results.append([(t,pc), True, d.get('name','No Name'),d.get('_channel','Unknown'),d.get('name','C0')[-2:]])
    #
    #         results.sort(key=lambda x:(x[4],x[2]))
    #         traces = [i[0] for i in results]
    #         userMark = [i[1] for i in results]
    #         names = [i[2] for i in results]
    #         devices = [i[3] for i in results]
    #         rearr_dataset = convert_list_to_X(traces),np.array(userMark),np.array(names),np.array(devices)
    #
    #         X, y, names,devices = removeDuplicates(*rearr_dataset)
    #         sorted_names = sort_datasets(names)
    #         idx_flag = [list(names).index(name) for name in sorted_names]
    #
    #         X = sort_by_idxflag(X, idx_flag)
    #         names = sort_by_idxflag(names, idx_flag)
    #         devices = sort_by_idxflag(devices, idx_flag)
    #         y = np.array([1] * len(X))
    #
    #         cutoffStart = 5
    #         cutoffEnd = 30
    #         normStart = 5
    #         normEnd = 10
    #
    #         t0 = time.perf_counter()
    #         print('Calculating...')
    #         smoothT = Pipeline([
    #             ('smooth', Smoother(stddev=2, windowlength=11, window='hanning')),
    #             ('normalize', Normalize(mode='mean', normalizeRange=(normStart, normEnd))),
    #             ('truncate', Truncate(cutoffStart=cutoffStart, cutoffEnd=cutoffEnd, n=90)),
    #             ('remove time', RemoveTime()),
    #         ])
    #         smoothed_X = smoothT.transform(X)
    #
    #         deriT = Pipeline([
    #             ('smooth', Smoother(stddev=2, windowlength=11, window='hanning')),
    #             ('normalize', Normalize(mode='mean', normalizeRange=(normStart, normEnd))),
    #             ('truncate', Truncate(cutoffStart=cutoffStart, cutoffEnd=cutoffEnd, n=90)),
    #             ('Derivitive', Derivitive(window=31, deg=3)),
    #             # ('remove time',RemoveTime()),
    #         ])
    #         deri_X = deriT.transform(X)
    #
    #
    #
    #         hCtT = Pipeline([
    #             ('smooth', Smoother(stddev=2, windowlength=11, window='hanning')),
    #             ('normalize', Normalize(mode='mean', normalizeRange=(normStart, normEnd))),
    #             ('truncate', Truncate(cutoffStart=cutoffStart, cutoffEnd=cutoffEnd, n=90)),
    #             ('Derivitive', Derivitive(window=31, deg=3)),
    #             ('peak', FindPeak()),
    #             ('logCt',HyperCt()),
    #
    #         ])
    #         hCtT_X = hCtT.transform(X)
    #         #
    #         # hCtTPredictT = Pipeline([
    #         #     ('smooth', Smoother(stddev=2, windowlength=11, window='hanning')),
    #         #     ('normalize', Normalize(mode='mean', normalizeRange=(normStart, normEnd))),
    #         #     ('truncate', Truncate(cutoffStart=cutoffStart, cutoffEnd=cutoffEnd, n=90)),
    #         #     ('Derivitive', Derivitive(window=31, deg=3)),
    #         #     ('peak', FindPeak()),
    #         #     ('logCt',HyperCt()),
    #         #     ('predictor',CtPredictor(ct=22,prominence=0.22,sd=0.05))
    #         # ])
    #         # hCtpred_X = hCtTPredictT.transform(X)
    #
    #
    #         hCtTPredictT = Pipeline([
    #             ('smooth', Smoother(stddev=2, windowlength=11, window='hanning')),
    #             ('normalize', Normalize(mode='mean', normalizeRange=(normStart, normEnd))),
    #             ('truncate', Truncate(cutoffStart=cutoffStart, cutoffEnd=cutoffEnd, n=90)),
    #             ('Derivitive', Derivitive(window=31, deg=3)),
    #             ('peak', FindPeak()),
    #             ('logCt',HyperCt()),
    #             ('predictor',SdPrPredictor(prominence=0.2,sd=0.106382))
    #         ])
    #
    #         hCtpred_X = hCtTPredictT.transform(X)
    #         print(f'Time taken to calculate {len(y)} data: {time.perf_counter()-t0:.3f} seconds.')
    #
    #         features = ['hyperCt', 'Pr', 'Sd5m']
    #
    #         with open(f'{picklefile}', 'a', newline='') as f:
    #             writer = csv.writer(f)
    #             if first_row:
    #                 writer.writerow(['Date', 'Name', 'Mark','Predict','Device']+features)
    #                 first_row = False
    #             for i, j in enumerate(y):
    #                 date = folder
    #                 name = names[i].strip()
    #                 hp_n = 'Positive' if hCtpred_X[i][0] else 'Negative'
    #                 data = list(hCtT_X[i])
    #                 writer.writerow([date, name, 'Positive' if j else 'Negative',hp_n,devices[i]] + [data[-1],data[1],data[5]])
    #         print(f"Write Ct and Prominence data to {picklefile}.")



# #%% Save scatter result a plot
# features =['left_ips',
#  'peak_prominence',
#  'peak_width',
#  'sdAtRightIps',
#  'sdAt3min',
#  'sdAt5min',
#  'sdAt10min',
#  'sdAt15min',
#  'sdAtEnd',
#  'hyperCt'
# ]

# # plot scatter plot of different features
# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# # axes = [i for j in axes for i in j]
# for (i, j), ax in zip(combinations([1,5,-1], 2), axes):
#     il = features[i]
#     jl = features[j]
#     ax.plot(hCtT_X[y == 0, i], hCtT_X[y == 0, j], 'gx', label='Negative')
#     ax.plot(hCtT_X[y == 1, i], hCtT_X[y == 1, j], 'r+', label='Positive')
#     ax.set_title(f'{il} vs {jl}')
#     ax.set_xlabel(il)
#     ax.set_ylabel(jl)
#     ax.legend()

# plt.tight_layout()
# fig.savefig(picklefile+'scatter.'+format,dpi=300)
# print(f"Feature Scatter plot is saved to {picklefile+'scatter.'+format}.")