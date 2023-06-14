#%% imports
from utils._util import ViewerDataSource
import matplotlib.pyplot as plt
import numpy as np
from utils.calling_algorithm import *
from utils.calling_algorithm import _version
from sklearn.pipeline import Pipeline
import textwrap
import csv
from itertools import combinations
import time
import json
from pathlib import Path

"""
Update Note:
2023/03/13: This is a special version for resolving the NTC issue of FluA primer sets
2023/01/31: Change the threshold to accomondate the new data structure
2022/11/17: Change the Ct definition to Ct = left_ips
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
picklefile = r"C:\Users\Public\Documents\SynologyDrive\Users\Sheng\SideProjects\20230316_CallingProblem\202303154ChannelFluBTest.picklez"
#%% user end input


if __name__ == '__main__':
    # picklefile = input('Enter picke file:\n').strip(' "')
    picklefiles_dir = input('Enter picke files directory:\n').strip(' "')
    picklefiles = [str(x) for x in Path(picklefiles_dir).glob('*.picklez')]
    savename = f'{picklefiles_dir}.json'
    # print(picklefiles)

#%% load data
# print(f'File you entered is: {picklefile}')
# print('reading data...')
cutoffEnds = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

hCtT_vs_cutoffEnd = {
    cutoffEnd: {
        'CT': [],
        'PR': [],
        'SD_3m': [],
        'SD_5m': [],
        'SD_10m': [],
        'SD_15m': [],
        'SD_End': [],
    }
    for cutoffEnd in cutoffEnds
}

for picklefile in picklefiles:
    #%%
    print(f'File you entered is: {picklefile}')
    dataSource = ViewerDataSource()
    pickleFiles = [picklefile]
    dataSource.load_picklefiles(pickleFiles)

    X, y, names,devices = removeDuplicates(*dataSource.exportXy())

    name_data_pairing = {
        name: [data]
        for name, data in zip(names, X)
    }

    #%%

    print('Total curve count is : '+str(len(X)))
    print("Total Positive Data: "+str(sum(y)))
    print("Total Negative Data: "+str(len(y)-sum(y)))

    #%% Calculate
    for cutoffEnd in cutoffEnds:

        #%%
        for name, X in name_data_pairing.items():
            print(f'Processing {name}...')

            cutoffStart = 8
            # cutoffEnd = 30
            normStart = cutoffStart
            normEnd = cutoffStart + 1

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
                ('Derivitive', Derivitive2(window=31, deg=3)),
                ('peak', FindPeak()),
                ('logCt',HyperCt()),
                
            ])
            hCtT_X = hCtT.transform(X)

            print(hCtT_X[0])

            hCtTPredictT = Pipeline([
                ('smooth', Smoother(stddev=2, windowlength=11, window='hanning')),
                ('normalize', Normalize(mode='mean', normalizeRange=(normStart, normEnd))),
                ('truncate', Truncate(cutoffStart=cutoffStart, cutoffEnd=cutoffEnd, n=90)),
                ('Derivitive', Derivitive2(window=31, deg=3)),
                ('peak', FindPeak()),
                ('logCt',HyperCt()),
                ('predictor',CtPredictor(ct=25,prominence=0,sd=0.1))
            ])
            hCtpred_X = hCtTPredictT.transform(X)

        # hCtT_vs_cutoffEnd[cutoffEnd] = {
        #     'CT': [x[0] for x in hCtT_X],
        #     'PR': [x[1] for x in hCtT_X],
        #     'SD_3m': [x[-9] for x in hCtT_X],
        #     'SD_5m': [x[-8] for x in hCtT_X],
        #     'SD_10m': [x[-7] for x in hCtT_X],
        #     'SD_15m': [x[-6] for x in hCtT_X],
        #     'SD_END' : [x[-5] for x in hCtT_X],
        # }
        # hCtTPredictT = Pipeline([
        #     ('smooth', Smoother(stddev=2, windowlength=11, window='hanning')),
        #     ('normalize', Normalize(mode='mean', normalizeRange=(normStart, normEnd))),
        #     ('truncate', Truncate(cutoffStart=cutoffStart, cutoffEnd=cutoffEnd, n=90)),
        #     ('Derivitive', Derivitive(window=31, deg=3)),
        #     ('peak', FindPeak()),
        #     ('logCt',HyperCt()),
        #     ('predictor',SdPrPredictor(prominence=0.2,sd=0.106382))
        # ])
        # hCtpred_X = hCtTPredictT.transform(X)
        #%%
            print(f'Time taken to calculate {len(y)} data: {time.perf_counter()-t0:.3f} seconds.')

            for x in hCtT_X:
                hCtT_vs_cutoffEnd[cutoffEnd]['CT'].append(x[0])
                hCtT_vs_cutoffEnd[cutoffEnd]['PR'].append(x[1])
                hCtT_vs_cutoffEnd[cutoffEnd]['SD_3m'].append(x[-9])
                hCtT_vs_cutoffEnd[cutoffEnd]['SD_5m'].append(x[-8])
                hCtT_vs_cutoffEnd[cutoffEnd]['SD_10m'].append(x[-7])
                hCtT_vs_cutoffEnd[cutoffEnd]['SD_15m'].append(x[-6])
                hCtT_vs_cutoffEnd[cutoffEnd]['SD_End'].append(x[-5])

# savename = picklefile.replace('.picklez', '_processed.json')
with open(savename, 'w') as f:
    json.dump(hCtT_vs_cutoffEnd, f, indent=4)
# #%% Plot data and save to svg file
# #############################################################################
# # plot the data                                                             #
# # overwrite column numbers; set to 0 to determine automatically             #
# #                                                                           #
# # ██████╗ ██╗      ██████╗ ████████╗██████╗  █████╗ ██████╗  █████╗         #
# # ██╔══██╗██║     ██╔═══██╗╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔══██╗        #
# # ██████╔╝██║     ██║   ██║   ██║   ██████╔╝███████║██████╔╝███████║        #
# # ██╔═══╝ ██║     ██║   ██║   ██║   ██╔═══╝ ██╔══██║██╔══██╗██╔══██║        #
# # ██║     ███████╗╚██████╔╝   ██║   ██║     ██║  ██║██║  ██║██║  ██║        #
# # ╚═╝     ╚══════╝ ╚═════╝    ╚═╝   ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝        #
# # This set how many columns in the figure, set to 0 automatically determine.#
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
# fig.savefig(f'{picklefile}_{_version}.{format}' ,dpi=300)
# print(f"Curve plot is saved to {picklefile}_{_version}.{format}.")



# #%% Save result to a csv file.
# features = ['hyperCt', 'Pr', 'Sd3m', 'Sd5m', 'Sd10m', 'Sd15m', 'SdEnd']

# # write result to csv file
# with open(f'{picklefile}_{_version}.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['Name', 'Mark','Predict','Device']+features)
#     for i, j in enumerate(y):
#         name = names[i].strip()
#         hp_n = 'Positive' if hCtpred_X[i][0] else 'Negative'
#         data = list(hCtT_X[i])
#         writer.writerow([name, 'Positive' if j else 'Negative',hp_n,devices[i]] + [data[-1],data[1],*data[4:9]])
# print(f"Write Ct and Prominence data to {picklefile+'.csv'}.")





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
