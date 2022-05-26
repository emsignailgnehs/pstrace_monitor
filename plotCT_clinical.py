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

    path = ui_getdirectory()
    filenames = [filename for filename in os.listdir(path) if filename.endswith('.picklez')]
    picklefiles = [f'{path}/{filename}' for filename in filenames]

    for filename, picklefile in zip(filenames, picklefiles):
        #%% load data
        print(f'File you entered is: {filename}')
        print('reading data...')
        dataSource = ViewerDataSource()
        pickleFiles = [picklefile]
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

        X = sort_by_idxflag(X, idx_flag)
        names = sort_by_idxflag(names, idx_flag)
        devices = sort_by_idxflag(devices, idx_flag)
        y = np.array([1] * len(X))


        # print('Total curve count is : '+str(len(X)))
        # print("Total Positive Data: "+str(sum(y)))
        # print("Total Negative Data: "+str(len(y)-sum(y)))

        #%% Calculate
        cutoffStart = 5
        cutoffEnd = 30
        normStart = 5
        normEnd = 10

        t0 = time.perf_counter()
        print('Calculating...')
        smoothT = Pipeline([
            ('smooth', Smoother(stddev=2, windowlength=11, window='hanning')),
            ('normalize', Normalize(mode='mean', normalizeRange=(normStart, normEnd))),
            ('truncate', Truncate(cutoffStart=cutoffStart, cutoffEnd=cutoffEnd, n=90)),
            ('remove time', RemoveTime()),
        ])
        smoothed_X = smoothT.transform(X)

        deriT = Pipeline([
            ('smooth', Smoother(stddev=2, windowlength=11, window='hanning')),
            ('normalize', Normalize(mode='mean', normalizeRange=(normStart, normEnd))),
            ('truncate', Truncate(cutoffStart=cutoffStart, cutoffEnd=cutoffEnd, n=90)),
            ('Derivitive', Derivitive(window=31, deg=3)),
            # ('remove time',RemoveTime()),
        ])
        deri_X = deriT.transform(X)



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


        #%% Plot data and save to svg file
        #############################################################################
        # plot the data                                                             #
        # overwrite column numbers; set to 0 to determine automatically             #
        #
                                               #
        # ██████╗ ██╗      ██████╗ ████████╗██████╗  █████╗ ██████╗  █████╗         #
        # ██╔══██╗██║     ██╔═══██╗╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔══██╗        #
        # ██████╔╝██║     ██║   ██║   ██║   ██████╔╝███████║██████╔╝███████║        #
        # ██╔═══╝ ██║     ██║   ██║   ██║   ██╔═══╝ ██╔══██║██╔══██╗██╔══██║        #
        # ██║     ███████╗╚██████╔╝   ██║   ██║     ██║  ██║██║  ██║██║  ██║        #
        # ╚═╝     ╚══════╝ ╚═════╝    ╚═╝   ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝        #
        # This set how many columns in the figure, set to 0 automatically determine.#
        # col = 4
        # # ymin and ymax is the min and max of y axis
        # ymin = 0.3
        # ymax = 1.3
        # format = 'svg'
        # #############################################################################


        # col = col or int(len(y)**0.5)
        # row = int(np.ceil(len(y) / col))
        # print(f'Generating curve plots in a {row} x {col} Grid')
        # fig, axes = plt.subplots(row, col, figsize=(col*4, row*3))
        # if row > 1:
        #     axes = [i for j in axes for i in j]

        # for i,j in enumerate(y):
        #     ax = axes[i]
        #     ax.set_ylim([0.3,1.5])

        #     smoothed_c = smoothed_X[i]
        #     t,deri,_ =  deri_X[i]
        #     left_ips,peak_prominence,peak_width, *sd= hCtT_X[i]

        #     curvePeakRange = findTimeVal(t,smoothed_c,left_ips,peak_width)
        #     xvals = np.linspace(t[0],t[-1],len(deri))


        #     # hyper ct
        #     hyperline = HyperCt.hyperF(None,hCtT_X[i][-4:-1])
        #     hyperCt = hCtT_X[i][-1]

        #     # plot smoothed current
        #     ax.plot(xvals,smoothed_c,color='red' if y[i] else 'green')
        #     # plot the signal drop part
        #     ax.plot(np.linspace(left_ips,left_ips+peak_width,len(curvePeakRange)) ,curvePeakRange,linewidth=4,alpha=0.75 )
        #     # plot plot the derivative peaks
        #     ax.plot(xvals,(deri - np.min(deri) ) / (np.max(deri) -np.min(deri) ) * (np.max(smoothed_c)-np.min(smoothed_c)) + np.min(smoothed_c),'--',alpha=0.8)
        #     # ax.plot(xvals,fitres(xvals),'b-.')
        #     # ax.plot(xvals,thresholdline(xvals),'b-.',alpha=0.7)
        #     # ax.plot([thresholdCt,thresholdCt],[0,2],'k-')

        #     # plot hyper fitting line
        #     ax.plot(xvals,hyperline(xvals),'k--',alpha=0.7)
        #     ax.plot([hyperCt,hyperCt],[min(smoothed_c),max(smoothed_c)],'k--',alpha=0.7)

        #     hp_n = '+' if hCtpred_X[i][0] else '-'
        #     m = '+' if y[i] else '-'
        #     title_color = 'red' if hCtpred_X[i][0]!=y[i] else 'green'

        #     ax.set_title(f'hCt:{hyperCt:.1f} Pm:{peak_prominence:.2f} SD5:{sd[2]:.4f} P:{hp_n} M:{m}',
        #     fontdict={'color':title_color,'fontsize':10})
        #     ax.set_xlabel('\n'.join(textwrap.wrap(
        #         names[i].strip(), width=45)), fontdict={'fontsize': 10})
        # plt.tight_layout()

        # # save to figure
        # fig.savefig(picklefile+'.'+format,dpi=300)
        # print(f"Curve plot is saved to {picklefile+'.'+format}.")



        #%% Save result to a csv file.
        features = ['Ct', 'Pr', 'Sd']

        # write result to csv file
        with open(f'{picklefile}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Mark','Predict','Device']+features)
            for i, j in enumerate(y):
                name = names[i].strip()
                hp_n = 'Positive' if hCtpred_X[i][0] else 'Negative'
                data = list(hCtT_X[i])
                writer.writerow([name, 'Positive' if j else 'Negative',hp_n,devices[i]] + [data[-1],data[1],data[5]])
        print(f"Write Ct and Prominence data to {picklefile+'.csv'}.")





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
