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
    #generalized naming: H NAS - 0163 - C1
    #                    |  |      |     |
    #                  cat type   id    ch
    def sorter_idnum(x):
        return x['name'].split('-')[1]
    def sorter_type(x):
        return x['name'].split('-')[0][1:]
    def sorter_cat(x):
        return x['name'].split('-')[0][0]
    def sorter_ch(x):
        return int(x['name'].split('-')[-1][-1])
    return sorted(datasets, key = lambda x: (sorter_idnum(x), sorter_type(x), sorter_cat(x), sorter_ch(x)))

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
idlist = set()
for dataset in sorted_datasets:
    name = dataset['name']
    cattype, id, ch = name.split('-')
    idlist.add(id)
    desc = dataset['desc']
    data = dataset['data']
    rearr_datasets[name] = {'desc': desc, 'data': data}

plot_query = []
cattypes = ['HNAS', 'PNAS', 'HSAL', 'PSAL']
for id in idlist:
    for cattype in cattypes:
        for ch in ['C1', 'C4']:
            plot_query.append(f'{cattype}-{id}-{ch}')

#%% plot the summary report
n_id = len(idlist)
w_ratio = [4,4,4,4]
h_ratio = [3,0.05,2,0.01]
fig= plt.figure(figsize= (sum(w_ratio), sum(h_ratio)*n_id))
gs = GridSpec(nrows= 4*n_id, ncols= 4, width_ratios= w_ratio, height_ratios= h_ratio*n_id)

for i, id in enumerate(sorted(idlist)):
    row_pc_vs_t = i * 4
    row_summary = i * 4 + 2
    for j, cattype in enumerate(cattypes):
        # plot individual channels
        ax_graph = fig.add_subplot(gs[row_pc_vs_t, j])
        try:
            for ch, clr in zip(['C1', 'C4'],['blue', 'red']):
                plot_query = f'{cattype}-{id}-{ch}'
                dat2plot = rearr_datasets[plot_query]['data']
                title = f'{cattype}-{id}'
                pc = [fit['pc'] for fit in dat2plot['fit']]
                time = np.linspace(0, 30-1/3, 90)[:len(pc)]
                ax_graph.plot(time, pc, linewidth = 2, color = clr, label = ch)
            ax_graph.legend()
        except KeyError:
            pass
        try:
            calling, summary = exp_summary(parse_desc(rearr_datasets[plot_query]['desc']))
            if calling == 'Positive':
                title_clr = 'red'
            elif calling == 'Negative':
                title_clr = 'green'
            else:
                title_clr = 'magenta'
            ax_graph.set_title(f'{title} ({calling})', color = title_clr)
            ax_graph.set_xlabel('time (min)')
            ax_graph.set_ylabel('Current ($\mu$A)')

            # fill in the summary
            ax_summary = fig.add_subplot(gs[row_summary, j])
            ax_summary.set_axis_off()
            ax_summary.text(0.05,0.5, summary, fontweight = 'bold')
        except KeyError:
            title = f'{cattype}-{id} (Incomplete/No Data)'
            ax_graph.set_title(title, color = 'gray')
            ax_graph.set_xlabel('time (min)')
            ax_graph.set_ylabel('Current ($\mu$A)')
            pass
plt.tight_layout()

savepath = picklefile.replace('.picklez', '')
fig.savefig(f'{savepath}_summary.svg')
