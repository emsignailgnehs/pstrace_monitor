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
import pandas as pd

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
def ui_getdirectory():
        root = tk.Tk() # pointing root to Tk() to use it as Tk() in program.
        root.withdraw() # Hides small tkinter window.
        root.attributes('-topmost', True) # Opened windows will be active. above all windows despite of selection.
        return filedialog.askdirectory(title= 'Please select the directory')

def sort_datasets(datasets):
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

def sort_by_idxflag(array, idx_flag):
    return np.array([array[i] for i in idx_flag])

if __name__ == '__main__':
    root_folder = r'C:\Users\Public\Documents\SynologyDrive\Projects\LAMP-Covid Sensor\EUA_Studies\Clinical_Study\ResultArchive'
    folders = [dir[1] for dir in os.walk(root_folder)][0][1:]
    folders = [folder for folder in folders if re.findall(r'\d{8}', folder)]
    for folder in folders:
        subfolder = f'{root_folder}\{folder}'
        files = [dir[2] for dir in os.walk(subfolder)][0]
        files = [file for file in files if re.findall(r'.*(_Clinical.*\.picklez)$', file)]
        for file in files:
            filepath = f'{subfolder}\{file}'
            #%% load data

            filepath = r'C:\Users\Public\Documents\SynologyDrive\Projects\LAMP-Covid Sensor\EUA_Studies\Clinical_Study\ResultArchive\20220513\20220513_control.picklez'

            print(f'File you entered is: {file}')
            print('reading data...')
            dataSource = ViewerDataSource()
            pickleFiles = [filepath]
            dataSource.load_picklefiles(pickleFiles)

            data = dataSource.rawView.get('data',[])
            results = []
            for d in data:
                t = timeseries_to_axis(d['data']['time'])
                pc = [i['pc'] for i in d['data']['fit']]
                # this '_channel' is actually the device name.
                results.append([(t,pc), True, d.get('name','No Name'),d.get('_channel','Unknown'),d.get('name','C0')[-2:]])

            results.sort(key=lambda x:(x[4],x[2]))
            traces = [i[0] for i in results]
            userMark = [i[1] for i in results]
            names = [i[2] for i in results]
            devices = [i[3] for i in results]
            rearr_dataset = convert_list_to_X(traces),np.array(userMark),np.array(names),np.array(devices)

            X, y, names,devices = removeDuplicates(*rearr_dataset)
            sorted_names = sort_datasets(names)
            idx_flag = [list(names).index(name) for name in sorted_names]

root = r'C:\Users\Public\Documents\SynologyDrive\Users\Sheng\SideProjects\20220325_ProcessingPCBdataFromTWXY\DataValidation_withTWXYreaders\20220719_DataValidation'
home_dir = f'{root}\parsed_home_calc'
twxy_dir = f'{root}\output'
files = [
    'HSAL-0258B-C4.txt',
    'PNAS-0193-C1.txt',
    'PSAL-0420-C4.txt'
]

def get_num(str_raw):
    pattern_num = r'\d+\.\d{6}'
    nums = re.findall(pattern_num, str_raw)
    return np.array([float(num) for num in nums])

home_datasets = {}
twxy_datasets = {}
for file in files:
    home_filename = f'{home_dir}\{file}'
    twxy_filename = f'{twxy_dir}\{file}'
    with open(home_filename, 'r') as f:
        home_datasets[file] = json.load(f)
    with open(twxy_filename, 'r') as f:
        twxy_raw = f.read()

    twxy_raw = ''.join(twxy_raw.split('\n'))
    norm_raw = twxy_raw.split('I/NO_TAG:db:normalize CH0')[-1].split('I/NO_TAG:db:DaoShu CH0')[0]
    daoshu_raw = twxy_raw.split('I/NO_TAG:db:normalize CH0')[-1].split('I/NO_TAG:db:DaoShu CH0')[-1].split('I/NO_TAG:findJiaoLeft')[0]
    norm = get_num(norm_raw)
    daoshu = get_num(daoshu_raw)

    x_1 = datasets[file]['pcs']
    x_0 = np.linspace(0, 30 - 1/3, 90)
    X = np.array([[np.array(x_0), np.array(x_1)]])

    cutoffStart = 5
    cutoffEnd = 30
    normStart = 5
    normEnd = 10

            # t0 = time.perf_counter()
            # print('Calculating...')
    smoothT = Pipeline([
        ('smooth', Smoother(stddev=2, windowlength=11, window='hanning')),
        ('normalize', Normalize(mode='mean', normalizeRange=(normStart, normEnd))),
        ('truncate', Truncate(cutoffStart=cutoffStart, cutoffEnd=cutoffEnd, n=90)),
        ('remove time', RemoveTime()),
    ])
    smoothed_X = smoothT.transform(X)

    plt.plot(x_0, smoothed_X[0], label = 'Home')
    plt.plot(x_0, norm, label = 'TWXY')
    plt.xlim([0, 30])
    plt.ylim([0.5, 1.05])
    plt.xlabel('Time (min)')
    plt.ylabel('Normalized pcs (arb.)')
    plt.legend()
    plt.show()

    deriT = Pipeline([
        ('smooth', Smoother(stddev=2, windowlength=11, window='hanning')),
        ('normalize', Normalize(mode='mean', normalizeRange=(normStart, normEnd))),
        ('truncate', Truncate(cutoffStart=cutoffStart, cutoffEnd=cutoffEnd, n=90)),
        ('Derivitive', Derivitive(window=31, deg=3)),
        # ('remove time',RemoveTime()),
    ])
    deri_X = deriT.transform(X)

    plt.plot(x_0, deri_X[0][1] * 100, label = 'Home')
    plt.plot(x_0, daoshu * 100, label = 'TWXY')
    plt.xlim([0, 30])
    plt.ylim([-0.05, 1.1])
    plt.xlabel('Time (min)')
    plt.ylabel('1$^{st}$ Derivative (arb.)')
    plt.legend()
    plt.show()

    opt_dict = {
        'original': X[0][1],
        'twxy_smooth': norm,
        'twxy_deri': daoshu,
        'home_smooth': smoothed_X[0],
        'home_deri': deri_X[0][1]
    }
    opt_df = pd.DataFrame(opt_dict)
    opt_df.to_csv(f'{root}\calc_opt_{file}.csv')

            hCtT = Pipeline([
                ('smooth', Smoother(stddev=2, windowlength=11, window='hanning')),
                ('normalize', Normalize(mode='mean', normalizeRange=(normStart, normEnd))),
                ('truncate', Truncate(cutoffStart=cutoffStart, cutoffEnd=cutoffEnd, n=90)),
                ('Derivitive', Derivitive(window=31, deg=3)),
                ('peak', FindPeak()),
                ('logCt',HyperCt()),

            ])
            hCtT_X = hCtT.transform(X)
            #
            # hCtTPredictT = Pipeline([
            #     ('smooth', Smoother(stddev=2, windowlength=11, window='hanning')),
            #     ('normalize', Normalize(mode='mean', normalizeRange=(normStart, normEnd))),
            #     ('truncate', Truncate(cutoffStart=cutoffStart, cutoffEnd=cutoffEnd, n=90)),
            #     ('Derivitive', Derivitive(window=31, deg=3)),
            #     ('peak', FindPeak()),
            #     ('logCt',HyperCt()),
            #     ('predictor',CtPredictor(ct=22,prominence=0.22,sd=0.05))
            # ])
            # hCtpred_X = hCtTPredictT.transform(X)


            hCtTPredictT = Pipeline([
                ('smooth', Smoother(stddev=2, windowlength=11, window='hanning')),
                ('normalize', Normalize(mode='mean', normalizeRange=(normStart, normEnd))),
                ('truncate', Truncate(cutoffStart=cutoffStart, cutoffEnd=cutoffEnd, n=90)),
                ('Derivitive', Derivitive(window=31, deg=3)),
                ('peak', FindPeak()),
                ('logCt',HyperCt()),
                ('predictor',SdPrPredictor(prominence=0.2,sd=0.106382))
            ])

            hCtpred_X = hCtTPredictT.transform(X)
            print(f'Time taken to calculate {len(y)} data: {time.perf_counter()-t0:.3f} seconds.')

            output_folder = r'C:\Users\Public\Documents\SynologyDrive\Projects\LAMP-Covid Sensor\EUA_Studies\Clinical_Study\ResultArchive\Home_calc'
            output_dict = {}
            for i, j in enumerate(y):
                name = names[i].strip()
                data = list(hCtT_X[i])
                pred = [data[-1],data[1],data[5]]
                output_dict[name] = {
                    'pcs': list(X[i][1]),
                    'ct': data[-1],
                    'pr': data[1],
                    'sd': data[5],
                    'deri': deri_X
                }
                savename = f'{output_folder}\{name}.txt'
                with open(savename, 'w') as f:
                    json.dump(output_dict[name],f)



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

# %%
