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
import json
from matplotlib.gridspec import GridSpec

def sort_datasets(datasets):
    #specific sorting function for clinical trial naming system
    #generalized naming: NTC - 11MAY2022 - C1
    #                    |       |         |
    #                  cat     date       ch
    # def sorter_idnum(x):
    #     return x['name'].split('-')[1]
    # def sorter_type(x):
    #     return x['name'].split('-')[0][1:]
    def sorter_readerid(x):
        return x['_id']
    def sorter_cat(x):
        return x['name'].split('-')[0]
    def sorter_ch(x):
        return int(x['name'].split('-')[-1][-1])
    return sorted(datasets, key = lambda x: (sorter_cat(x), sorter_readerid(x), sorter_ch(x)))

def parse_desc(desc):
    meta, exp, calling = [json.loads(e) for e in desc.split("|")[1:]]
    meta['created'] = ', '.join([e.split('.')[0] for e in meta['created'].split('T')])
    dict_desc = {
        'meta': meta,
        'exp': exp,
        'calling': calling
    }
    return dict_desc

def exp_summary(dict_desc):
    calling = dict_desc['calling']
    date = dict_desc['meta']['created']
    reader = dict_desc['meta']['deviceId']
    channelResults = [e.strip() for e in dict_desc['exp']['channelResult'].split(",")]
    temp = dict_desc['exp']['avgTemp']

    opt_msg1 = f"Reader: {reader}\n\n"
    opt_msg2 = f"{channelResults[0]}\n{channelResults[1]}\n\nAvg. Temp.: {temp}\nDate: {date}"
    return calling, opt_msg1 + opt_msg2

#%% user end input
if __name__ == '__main__':
    picklefile = input('Enter picke file:\n').strip(' "')

#%% load data
print(f'File you entered is: {picklefile}')
print('reading data...')
dataSource = ViewerDataSource()
pickleFiles = [picklefile]
dataSource.load_picklefiles(pickleFiles)

datasets = dataSource.rawView['data']
#%% rearrange datasets to facilitate preprocessing & generate a list of ids
sorted_datasets = sort_datasets(datasets)
rearr_datasets = {}
for dataset in datasets:
    cattype = dataset['name'].split('-')[0]
    id = dataset['_id']
    rearr_datasets[f'{cattype}-{id}'] = {}
for dataset in sorted_datasets:
    cattype = dataset['name'].split('-')[0]
    ch = dataset['name'].split('-')[-1]
    id = dataset['_id']
    desc = dataset['desc']
    data = dataset['data']
    rearr_datasets[f'{cattype}-{id}'][ch] = {'desc': desc, 'data': data}

column_num = 4
row_num = len(rearr_datasets) // 4
if len(rearr_datasets) % 4:
    row_num += 1
w_ratio = [4,4,4,4]
h_ratio = [3,0.05,2,0.01]
fig= plt.figure(figsize= (sum(w_ratio), sum(h_ratio)*row_num))
gs = GridSpec(nrows= 4*row_num, ncols= column_num, width_ratios= w_ratio, height_ratios= h_ratio*row_num)

for idx, file_id in enumerate(list(rearr_datasets.keys())):
    col = idx % 4
    i = idx // 4
    row_pc_vs_t = i * 4
    row_summary = i * 4 + 2
    cattype, id = file_id.split('-')

    ax_graph = fig.add_subplot(gs[row_pc_vs_t, col])
    try:
        for ch, clr in zip(['C1', 'C4'],['blue', 'red']):
            dat2plot = rearr_datasets[file_id][ch]['data']
            title = f'{cattype}'
            pc = [fit['pc'] for fit in dat2plot['fit']]
            time = np.linspace(0, 30-1/3, 90)[:len(pc)]
            ax_graph.plot(time, pc, linewidth = 2, color = clr, label = ch)
        ax_graph.legend()
    except KeyError:
        pass
    try:
        calling, summary = exp_summary(parse_desc(rearr_datasets[file_id][ch]['desc']))
        if calling == 'Positive':
            title_clr = 'red'
        elif calling == 'Negative':
            title_clr = 'green'
        else:
            title_clr = 'magenta'
        ax_graph.set_title(f'{title} ({calling})', color = title_clr)
        ax_graph.set_ylim([0, 30])
        ax_graph.set_xlabel('time (min)')
        ax_graph.set_ylabel('Current ($\mu$A)')

        # fill in the summary
        ax_summary = fig.add_subplot(gs[row_summary, col])
        ax_summary.set_axis_off()
        ax_summary.text(0.05,0.5, summary, fontweight = 'bold')
    except KeyError:
        title = f'{cattype} (Incomplete/No Data)'
        ax_graph.set_title(title, color = 'gray')
        ax_graph.set_xlabel('time (min)')
        ax_graph.set_ylabel('Current ($\mu$A)')
        pass
plt.tight_layout()

savepath = picklefile.replace('.picklez', '')
fig.savefig(f'{savepath}_ControlSummary.svg')
