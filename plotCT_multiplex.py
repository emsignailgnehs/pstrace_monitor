#%%
"""Import the necessary packages"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
import textwrap
import csv
from itertools import combinations
import time
from pathlib import Path
import json
from tkinter import Tk, simpledialog     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import re
import matplotlib.gridspec as gridspec
import logging

from utils._util import ViewerDataSource
from utils.calling_algorithm import *
from utils.calling_algorithm import _version

root = Path(__file__).parent

#%%
"""Get config file and pickle file locations with a GUI"""
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
# show an "Open" dialog box and return the path to the selected file
config_file = askopenfilename(
    title= 'open config file',
    filetypes=[('json files', '*.json')],
    initialdir= root
)
print(f'config file location:\n\t{config_file}\n')
print('-' * 40)

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
# show an "Open" dialog box and return the path to the selected file
picklefile = askopenfilename(
    title= 'open .picklez file',
    filetypes=[('picklez files', '*.picklez')],
    initialdir= root
)
print(f'pickle file location:\n\t{picklefile}\n')
print('-' * 40)

# config_file = r"C:\Users\Public\Documents\SynologyDrive\Users\Sheng\SideProjects\pstrace_monitor\plotCT_multiplex\config.json"
# picklefile = r"c:\Users\Public\Documents\SynologyDrive\Users\Sheng\SideProjects\20230727-AlgorithmTraining\FluA-NTC.picklez"

#%%
"""Load config file"""
with open(config_file) as f:
    config = json.load(f)
assay_config = config['assay_config']
ch_config = config['ch_config']

"""Load pickle file"""
print(f'File you entered is: {picklefile}')
print('reading data...')
dataSource = ViewerDataSource()
pickleFiles = [picklefile]
dataSource.load_picklefiles(pickleFiles)

X, user_marks, names,devices = removeDuplicates(*dataSource.exportXy())

print('Total curve count is : '+str(len(X)))
print("Total Positive Data: "+str(sum(user_marks)))
print("Total Negative Data: "+str(len(user_marks)-sum(user_marks)))

#%%
"""Extract assay configuration and convert the configuration into algroithm parameters
"""
analysis_length = assay_config['analysis_length']
preheat_length = assay_config['preheat_length']
cutoff_length = assay_config['cutoff_length']

cutoffStart = preheat_length + cutoff_length # remove preheat data and part of analysis data
cutoffEnd = preheat_length + analysis_length
normStart = cutoffStart
normEnd = cutoffStart + assay_config['normalization_length']

#%%
"""Group the data into channels for channel specific calling
"""
channel_data = {}
ch_pattern = r'\-(C\d)'
for name, datum, user_mark, device in zip(names, X, user_marks, devices):
    ch = re.findall(ch_pattern, name)[0]
    if ch not in channel_data.keys():
        channel_data[ch] = {
            'name': [],
            'user_mark': [],
            'device': [],
            'rawdata': [],
            'smooth': None,
            'deri': None,
            'hCt': None,
            'hCtpred': None,
        }
    channel_data[ch]['name'].append(name)
    channel_data[ch]['rawdata'].append(datum)
    channel_data[ch]['user_mark'].append(user_mark)
    channel_data[ch]['device'].append(device)

#%%
"""Configure the output file names"""
output_dir = picklefile.replace('.picklez', '_output')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

filename = Path(picklefile).stem

logger_location = os.path.join(output_dir, 'log.log')
output_csv_name = os.path.join(output_dir, f'CALLING-{filename}.csv')
output_ch_figure_names = {
    ch:os.path.join(output_dir, f'FIG-{ch}-{filename}.png')
    for ch in channel_data.keys()
}

#%%
"""Set up logger"""
logging.basicConfig(
    filename=logger_location,
    level=logging.INFO,
    format='[%(asctime)s][%(name)s]: %(message)s',
)
logger = logging.getLogger(Path(__file__).stem)
logger.info(f'pickle file: {picklefile}')
logger.info(f'config file: {config_file}')
logger.info('-'*50)
for ch in channel_data.keys():
    ch_curve_count = len(channel_data[ch]['rawdata'])
    ch_pos_count = sum(channel_data[ch]['user_mark'])
    ch_neg_count = ch_curve_count - ch_pos_count
    logger.info(f'channel {ch}: {len(channel_data[ch]["rawdata"])} curves (+: {ch_pos_count}; -: {ch_neg_count})')
logger.info('='*50)
logger.info(f'analysis_length: {analysis_length}')
logger.info(f'preheat_length: {preheat_length}')
logger.info('-'*50)
logger.info(f'cutoff: {cutoffStart} -> {cutoffEnd}')
logger.info(f'normalization: {normStart} -> {normEnd}')
logger.info('='*50)
#%%
"""Channel specific calling"""
skipped_channels = []
for ch, data in channel_data.items():
    try:
        target, pr, ct, sd = ch_config[ch].values()
    except KeyError:
        skipped_channels.append(ch)
        print(f'Channel {ch} is not in the config file, skip')
        logger.info(f'Channel {ch} is not in the config file, skip')
        continue

    X = np.array(data['rawdata'])
    print(f'Calculating {ch} ({target}) data...')
    # normT = Pipeline([
    #     ('normalize', Normalize(mode='mean', normalizeRange=(normStart, normEnd)))
    # ])
    # channel_data[ch]['norm'] = normT.transform(X)

    smoothT = Pipeline([
        ('smooth', Smoother(stddev=2, windowlength=11, window='hanning')),
        ('normalize', Normalize(mode='mean', normalizeRange=(normStart, normEnd))),
        # ('truncate', Truncate(cutoffStart=cutoffStart, cutoffEnd=cutoffEnd, n=90)),
        # ('remove time', RemoveTime()),
    ])
    channel_data[ch]['smooth'] = smoothT.transform(X)

    deriT = Pipeline([
        ('smooth', Smoother(stddev=2, windowlength=11, window='hanning')),
        ('normalize', Normalize(mode='mean', normalizeRange=(normStart, normEnd))),
        ('truncate', Truncate(cutoffStart=cutoffStart, cutoffEnd=cutoffEnd, n=90)),
        ('Derivitive', Derivitive2(window=31, deg=3)),
        # ('remove time',RemoveTime()),
    ])
    channel_data[ch]['deri'] = deriT.transform(X)

    hCtT = Pipeline([
        ('smooth', Smoother(stddev=2, windowlength=11, window='hanning')),
        ('normalize', Normalize(mode='mean', normalizeRange=(normStart, normEnd))),
        ('truncate', Truncate(cutoffStart=cutoffStart, cutoffEnd=cutoffEnd, n=90)),
        ('Derivitive', Derivitive2(window=31, deg=3)),
        ('peak', FindPeak()),
        ('logCt',HyperCt()),
        
    ])
    channel_data[ch]['hCt'] = hCtT.transform(X)

    hCtTPredictT = Pipeline([
        ('smooth', Smoother(stddev=2, windowlength=11, window='hanning')),
        ('normalize', Normalize(mode='mean', normalizeRange=(normStart, normEnd))),
        ('truncate', Truncate(cutoffStart=cutoffStart, cutoffEnd=cutoffEnd, n=90)),
        ('Derivitive', Derivitive2(window=31, deg=3)),
        ('peak', FindPeak()),
        ('logCt',HyperCt()),
        ('predictor',CtPredictor(ct=ct,prominence=pr,sd=sd))
    ])
    channel_data[ch]['hCtpred'] = hCtTPredictT.transform(X)

#%%
"""Output the calling result into a csv file
1. print assay configuration
2. print channel configuration
"""
with open(output_csv_name, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # print the skipped channels
    if skipped_channels:
        writer.writerow(['Skipped channels:', *skipped_channels])
        writer.writerow([])
    # print the assay configuration
    writer.writerow(['analyzed', f'{cutoffStart} - {cutoffEnd} min'])
    writer.writerow(['norm', f'{normStart} - {normEnd} min'])
    writer.writerow([])
    # print the channel configuration
    for ch, data in channel_data.items():
        target, pr, ct, sd = ch_config[ch].values()

        names = data['name']
        user_marks = data['user_mark']
        devices = data['device']
        hCtpred = data['hCtpred']
        peakwidths = data['hCt']

        writer.writerow([f'{ch}:{target}','','','',f'ct < {ct}', f'sd > {sd}'])
        writer.writerow(['Name', 'Mark', 'Predict', 'Device', 'CT', 'SD5', 'SD10', 'SDE'])
        for name, user_mark, device, pred, peakwidth in zip(names, user_marks, devices, hCtpred, peakwidths):
            prediction = pred['prediction']
            ct = pred['ct']
            sd5 = pred['sdAt5min']
            sd10 = pred['sdAt10min']
            sdEnd = pred['sdAtEnd']
            writer.writerow([
                name, 
                'Positive' if user_mark else 'Negative', 
                'Positive' if prediction else 'Negative',
                device,
                ct,
                sd5,
                sd10,
                sdEnd
            ])
        writer.writerow(['============', '============', '============', '============', '============', '============', '============', '============'])
        writer.writerow([])
        writer.writerow([])
logger.info(f'Output csv file finished')
#%%
"""Output the calling result visualization
key parameters: infleciton point, signal drop time, calling result
"""
def normalize_to_refdata(array, refdata):
    # normalize the array
    array = (array - array.min()) / (array.max() - array.min())
    # scale the array to the refdata
    array = array * (refdata.max() - refdata.min()) + refdata.min()
    return array

def find_norm_range(twod_array, normStart, normEnd):
    # find the index of the normStart and normEnd
    normStartIndex = np.argmin(np.abs(np.array(twod_array[0]) - normStart))
    normEndIndex = np.argmin(np.abs(np.array(twod_array[0]) - normEnd))
    if normStartIndex == normEndIndex:
        normEndIndex += 1
    return normStartIndex, normEndIndex

def findTimeVal(t,val,t0,dt,lower_bound,upper_bound):
    t = np.array(t)
    val = np.array(val)
    range_max = max(t0, lower_bound)
    range_min = min(t0 + dt, upper_bound)
    range_flag = (t >= range_max) * (t <= range_min)
    return t[range_flag], val[range_flag]

#%%
for ch in channel_data.keys():
    """Set up the data structure for plotting
    """
    data = channel_data[ch]
    rs = data['rawdata']
    ss = data['smooth']
    ds = [
        [
            np.linspace(d[0][0], d[0][-1], len(d[1])),
            d[1]
        ]
        for d in data['deri']
    ]
    hCt = data['hCt']
    hCtpred = data['hCtpred']
    names = data['name']
    user_marks = data['user_mark']
    """Set up the outer grid to arrage the plotted reuslts
    """
    col = 4
    row = int(np.ceil(len(rs) / col))
    # set resolution for the plot
    plt.rcParams['figure.dpi'] = 150
    # create a gridspec for plots
    fig = plt.figure(figsize=(col*6, row*8))
    outer_gs = gridspec.GridSpec(row, col, wspace=0.2, hspace=0.2)
    fig.suptitle(
        f"""{ch}: {ch_config[ch]['target']}  =>  CT={ch_config[ch]['CT']}, SD={ch_config[ch]['SD']}""",
        fontsize=25,
        fontweight='bold',
        y=0.95
    )

    for i,(r,s,d,name,user_mark, calc, pred) in enumerate(zip(rs,ss,ds,names,user_marks, hCt, hCtpred)):
        normStartIndex, normEndIndex = find_norm_range(s, normStart, normEnd)
        left_ips = calc[0]['left_ips']
        peak_width = calc[0]['peak_width']
        curvePeakRange = findTimeVal(s[0], s[1], left_ips, 10, lower_bound=cutoffStart, upper_bound= cutoffEnd)

        inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[i], wspace=0.1, hspace=0.1)
        ax1 = plt.Subplot(fig, inner_gs[0])
        ax2 = plt.Subplot(fig, inner_gs[1])
        # fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(6,8))
        ax1.plot(r[0],r[1],label=f'{ch} raw', color= 'black', linewidth=2)
        ax2.plot(s[0],s[1],label='smoothed', color= 'red' if user_mark else 'green', linewidth=2)
        ax2.plot(d[0],normalize_to_refdata(d[1], s[1]),label='deri', color= 'orange', linewidth=2, linestyle='--')
        if normEndIndex - normStartIndex <=1:
            ax2.scatter(s[0][normStartIndex:normEndIndex],s[1][normStartIndex:normEndIndex],label='norm point', color= 'cyan')
        else:
            ax2.plot(s[0][normStartIndex:normEndIndex],s[1][normStartIndex:normEndIndex],label='norm range', color= 'cyan', linewidth=2, linestyle='-')
        ax2.plot(
            curvePeakRange[0], curvePeakRange[1]
            , color= 'blue',linewidth=8,alpha=0.2, label='SD range'
        )
        ax2.axvline(x= cutoffStart, color='black', linestyle='--', linewidth=0.5)
        ax2.axvline(x= cutoffEnd, color='black', linestyle='--', linewidth=0.5)
        ax1.set_title(name, fontweight= 'bold')
        ax2.set_xlabel(f'{"PASSED" if user_mark == pred["prediction"] else "FAILED"} (User mark: {user_mark}; Predicted: {pred["prediction"]})\nCT={pred["ct"]:.3f}; SD 10={pred["sdAt10min"]:.3f}',
                    fontweight= 'bold',
                    color= 'blue' if user_mark == pred['prediction'] else 'red'
                    )
        ax1.legend(title = '<Raw Data>')
        ax2.legend(title = '<Processed>')
        fig.add_subplot(ax1)
        fig.add_subplot(ax2)
    plt.tight_layout()
    fig.savefig(output_ch_figure_names[ch])
    logger.info(f'Output figure for {ch} finished')

#%%
