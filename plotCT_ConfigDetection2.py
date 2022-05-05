cwd = r'C:\Users\wenta\Desktop\pstrace_monitor-master'
import sys
sys.path.append(cwd)
from utils._util import ViewerDataSource
import matplotlib.pyplot as plt
import numpy as np
from utils.calling_algorithm import *
from sklearn.pipeline import Pipeline
import textwrap
import csv
from itertools import combinations
import re
import pandas as pd

"""
2021/11/19 Modification:
- Expanding the functionality that could resolve different configuration of sensors
  Current:
        1. Dual RP4
        2. Dual N7
        3. N7/RP4

2022/02/11 Modification:
- Adding automated data sorting
    For data with name end with experiment id such as (MC 2-1),
    wehre 2 is the experiment number and 1 is the sequence,
    the algorithm will automatically sort the data numerically

- For heterogeneous channel sensors, the sorting algorithm can automatically
  separate C1 and C4

2022/03/16 Modification:
- Found that the MacOS input path in terminal would show blank " " as "\\"
  which confuses the program
    Add a function to parse the filename before proceeding
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

#################### FUNCTIONS #######################################
def sensor_configuration():
    sensor_config = input('Enter the number of the configuration:\n(1): Dual RP4; (2) Dual N7; (3) N7/RP4\n')
    return int(sensor_config)

def hCtTpredictT_var(cycle_time, prom, signal_drop):
    return Pipeline([
        ('smooth', Smoother(stddev=2, windowlength=11, window='hanning')),
        ('normalize', Normalize(mode='mean', normalizeRange=(normStart, normEnd))),
        ('truncate', Truncate(cutoffStart=cutoffStart, cutoffEnd=cutoffEnd, n=90)),
        ('Derivitive', Derivitive(window=31, deg=3)),
        ('peak', FindPeak()),
        ('logCt',HyperCt()),
        ('predictor',SdPrPredictor(prominence = prom, sd = signal_drop))
    ])

def re_search(pattern,string):
    exp = re.compile(pattern)
    result = exp.search(string)
    return(result.group())

def file_id_gen(string):
    #used to parse the conventional naming into a set of id
    #for sorting automation purposes
    #conventional naming: .....(MC 1-1)
    raw = [x.strip() for x in string.split(')')]
    channel = int(raw[-1].strip('-C'))
    exp_id = [int(x.strip()) for x in re_search(r'\d.*\d',raw[0]).split('-')]
    exp_id.append(channel)
    return exp_id

def heterochannel_sort(names):
    file_ids = []
    for name in names:
        string = re_search(r'\(.*\S{2}.*\d.*\d.*\).*\-\S\d', name)
        file_ids.append(file_id_gen(string))
    df = pd.DataFrame(file_ids, columns = ['exp', 'seq', 'ch'])
    df['ogidx'] = [i for i in range(len(names))]

    groups = {}
    for exp, group in df.groupby('exp'):
        groups['exp' + str(exp)] = {}
        for ch, sec_group in group.groupby('ch'):
            groups['exp' + str(exp)]['ch' + str(ch)] = sec_group.sort_values(by = 'seq')

    ogidx = []
    for key in groups.keys():
        for sub_key in groups[key].keys():
            for element in groups[key][sub_key]['ogidx']:
                ogidx.append(element)
    return ogidx

def homochannel_sort(names):
    file_ids = []
    for name in names:
        string = re_search(r'\(.*\S{2}.*\d.*\d.*\).*\-\S\d', name)
        file_ids.append(file_id_gen(string))
    df = pd.DataFrame(file_ids, columns = ['exp', 'seq', 'ch'])
    df['ogidx'] = [i for i in range(len(names))]

    groups = {}
    for exp, group in df.groupby('exp'):
        groups['exp' + str(exp)] = {}
        for seq, sec_group in group.groupby('seq'):
            groups['exp' + str(exp)]['seq' + str(seq)] = sec_group.sort_values(by = 'ch')

    ogidx = []
    for key in groups.keys():
        for sub_key in groups[key].keys():
            for element in groups[key][sub_key]['ogidx']:
                ogidx.append(element)
    return ogidx
################### INPUT PARAMETERS #############################################
picklefile = r"C:\Users\Public\Documents\SynologyDrive\Projects\LAMP-Covid Sensor\Data Export\20220211\20220211 SL laminate pressure test and stored vs fresh buffer.picklez"

"""
thresholds format: [Ct, Prominence, SD]
"""
thresholds = {
'RP4': [26.769231, 0.2, 0.106382],
'N7': [24.263158, 0.2, 0.106382]
}

################### MAIN #############################################
if __name__ == '__main__':

    sensor_config = 0
    while not sensor_config in [1, 2, 3]:
        sensor_config = sensor_configuration()
        print('===============================\n')
        if not sensor_config in [1, 2, 3]:
            print('Please enter a NUMBER 1, 2, or 3 !!!\n')

    picklefile = input('Enter picke file:\n').strip(' "')

filename = picklefile.split("\\")[-1]
if len(filename.split("//")) > 1:
    filename = " ".join(filename.split("//"))
    print('1111')
    picklefile = "\\".join(picklefile.split("\\")[:-2] + [filename])
    print('2222')

print(f'File you entered is: {filename}')
print(picklefile)
print('reading data...')
dataSource = ViewerDataSource()
pickleFiles = [picklefile]
dataSource.load_picklefiles(pickleFiles)

X, y, names,devices = removeDuplicates(*dataSource.exportXy())

C1_flag = ['-C1' in name for name in names]
C4_flag = ['-C4' in name for name in names]

if sensor_config == 1:
    C1_threshold = thresholds['RP4']
    C4_threshold = thresholds['RP4']
    sorting_idx = homochannel_sort(names)
elif sensor_config == 2:
    C1_threshold = thresholds['N7']
    C4_threshold = thresholds['N7']
    sorting_idx = homochannel_sort(names)
else:
    C1_threshold = thresholds['N7']
    C4_threshold = thresholds['RP4']
    sorting_idx = heterochannel_sort(names)

print('Total curve count is : '+str(len(X)))
print("Total Positive Data: "+str(sum(y)))
print("Total Negative Data: "+str(len(y)-sum(y)))

cutoffStart = 5
cutoffEnd = 30
normStart = 5
normEnd = 10

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

"""
1. Split X into two arrays according to their corresponding channels
2. Apply two thresholds to C1 and C4 to generate hCtpred_X_Channel
3. Stitch the two array together to generate hCtpred_X
"""

ct_C1, pr_C1, sd_C1 = C1_threshold
ct_C4, pr_C4, sd_C4 = C4_threshold

X_C1 = X[C1_flag]
X_C4 = X[C4_flag]
hCtpred_X_C1 = hCtTpredictT_var(ct_C1, pr_C1, sd_C1).transform(X_C1)
hCtpred_X_C4 = hCtTpredictT_var(ct_C4, pr_C4, sd_C4).transform(X_C4)

set_size = np.shape(hCtpred_X_C1)[1]
sets_num = len(X)

hCtpred_X = np.empty((sets_num, set_size))
hCtpred_X[C1_flag] = hCtpred_X_C1
hCtpred_X[C4_flag] = hCtpred_X_C4

#############################################################################
# plot the data                                                             #
# overwrite column numbers; set to 0 to determine automatically             #
#                                                                           #
# ██████╗ ██╗      ██████╗ ████████╗██████╗  █████╗ ██████╗  █████╗         #
# ██╔══██╗██║     ██╔═══██╗╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔══██╗        #
# ██████╔╝██║     ██║   ██║   ██║   ██████╔╝███████║██████╔╝███████║        #
# ██╔═══╝ ██║     ██║   ██║   ██║   ██╔═══╝ ██╔══██║██╔══██╗██╔══██║        #
# ██║     ███████╗╚██████╔╝   ██║   ██║     ██║  ██║██║  ██║██║  ██║        #
# ╚═╝     ╚══════╝ ╚═════╝    ╚═╝   ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝        #
# This set how many columns in the figure, set to 0 automatically determine.#
col = 4
# ymin and ymax is the min and max of y axis
ymin = 0.3
ymax = 1.3
format = 'svg'
#############################################################################


col = col or int(len(y)**0.5)
row = int(np.ceil(len(y) / col))
print(f'Generating curve plots in a {row} x {col} Grid')
fig, axes = plt.subplots(row, col, figsize=(col*4, row*3))
if row > 1:
    axes = [i for j in axes for i in j]

for fig_num, idx in enumerate(sorting_idx):
    i = idx
    ax = axes[fig_num]
    ax.set_ylim([0.4,1.3])

    smoothed_c = smoothed_X[i]
    t,deri,_ =  deri_X[i]
    left_ips,peak_prominence,peak_width, *sd= hCtT_X[i]

    curvePeakRange = findTimeVal(t,smoothed_c,left_ips,peak_width)
    xvals = np.linspace(t[0],t[-1],len(deri))

    # hyper ct
    hyperline = HyperCt.hyperF(None,hCtT_X[i][-4:-1])
    hyperCt = hCtT_X[i][-1]

    # plot smoothed current
    ax.plot(xvals,smoothed_c,color='red' if y[i] else 'green')
    # plot the signal drop part
    ax.plot(np.linspace(left_ips,left_ips+peak_width,len(curvePeakRange)) ,curvePeakRange,linewidth=4,alpha=0.75 )
    # plot plot the derivative peaks
    ax.plot(xvals,(deri - np.min(deri) ) / (np.max(deri) -np.min(deri) ) * (np.max(smoothed_c)-np.min(smoothed_c)) + np.min(smoothed_c),'--',alpha=0.8)
    # ax.plot(xvals,fitres(xvals),'b-.')
    # ax.plot(xvals,thresholdline(xvals),'b-.',alpha=0.7)
    # ax.plot([thresholdCt,thresholdCt],[0,2],'k-')

    # plot hyper fitting line
    ax.plot(xvals,hyperline(xvals),'k--',alpha=0.7)
    ax.plot([hyperCt,hyperCt],[0,2],'k--',alpha=0.7)

    hp_n = '+' if hCtpred_X[i][0] else '-'
    m = '+' if y[i] else '-'
    title_color = 'red' if hCtpred_X[i][0]!=y[i] else 'green'

    ax.set_title(f'hCt:{hyperCt:.1f} Pm:{peak_prominence:.2f} SD5:{sd[2]:.4f} P:{hp_n} M:{m}',
    fontdict={'color':title_color,'fontsize':10})
    ax.set_xlabel('\n'.join(textwrap.wrap(
        names[i].strip(), width=45)), fontdict={'fontsize': 10})
plt.tight_layout()

# save to figure
fig.savefig(picklefile+'.'+format,dpi=300)
print(f"Curve plot is saved to {picklefile+'.'+format}.")


features = ['hyperCt', 'Prominence', 'SD_5min']

# write result to csv file
with open(f'{picklefile}.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'User Mark','Prediction','Device']+features)
    for idx in sorting_idx:
        i = idx
        j = y[idx]
        # print(i,',',j)
        name = names[i].strip()
        hp_n = 'Positive' if hCtpred_X[i][0] else 'Negative'
        data = list(hCtT_X[i])
        writer.writerow([name, 'Positive' if j else 'Negative',hp_n,devices[i]] + [data[-1],data[1],data[5]])
print(f"Write Ct and Prominence data to {picklefile+'.csv'}.")


# # plot scatter plot of different features
# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# # axes = [i for j in axes for i in j]
# for (i, j), ax in zip(combinations([1,5,-1], 2), axes):
#     il = features[i]
#     jl = features[j]
#     ax.plot(tCt_X[y == 0, i], tCt_X[y == 0, j], 'gx', label='Negative')
#     ax.plot(tCt_X[y == 1, i], tCt_X[y == 1, j], 'r+', label='Positive')
#     ax.set_title(f'{il} vs {jl}')
#     ax.set_xlabel(il)
#     ax.set_ylabel(jl)
#     ax.legend()

# plt.tight_layout()
# fig.savefig(picklefile+'scatter.'+format,dpi=300)
# print(f"Feature Scatter plot is saved to {picklefile+'scatter.'+format}.")
