import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import re
from matplotlib.patches import Rectangle
plt.rcParams['figure.dpi'] = 150

msg = ' | {"name": "20220411 N6O4/R18S 40 copy BioIVT (SL 1-3)", "deviceId": "AMS-ADM", "deviceSerial": "000005800170", "deviceDataId": null, "chipType": 4, "created": "2022-03-25T21:19:28.740427", "device": "AMS-ADM", "exp": "", "desc": ""} | {"stat": "ok", "avgTemp": "67.53 C", "chipInsertion": "29 / 29", "fluid": "C1-14/14, C4-14/14", "channelResult": "C1-Negative Ct:30.0 Pr:0.1 Sd:0.02,C4-Positive Ct:21.0 Pr:0.6 Sd:0.14"} | "Negative"'

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

def print_dict(dict):
    keys = list(dict.keys())
    for key in keys:
        print(key, ': ', dict[key], '\n')

calling, summary = exp_summary(parse_desc(msg))

sample_data = {
    'data': [x, y],

}

dict = {
    'HSAL-1': [x, y],
    'PSAL-1': [x, y],
    'HNAS-1': [x, y],
    'PNAS-1': [x, y],
    'HSAL-2': [x, y],
    'PSAL-2': [x, y],
    'HNAS-2': [x, y],
    'PNAS-2': [x, y]
}

try:
    k = dict['HNAS-3']
except KeyError:
    k = None

list = [0, 1, 2, 3]

list2 = list * 3

list2

fig = plt.figure(figsize = (16, 5.3))
gs = GridSpec(nrows=4, ncols=4, width_ratios=[1, 1, 1, 1], height_ratios=[3, 0.1, 2, 0.2])

test, axes = plt.subplots(3, 2)

axes

for row in axes:
    for ax in row:
        ax

time = [0, 1]
height = [0, 1]
weight = [0, 1]
# First axes
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(time, height)
# ax0.axes.xaxis.set_visible(False)
# ax0.axes.yaxis.set_visible(False)
calling = 'Invalid'
clr = 'red'
if calling == 'Negative':
    clr = 'green'
if calling == 'Invalid':
    clr = 'magenta'


ax0.set_title(f"HSAL-011 ({calling})", fontsize = 16, fontweight = 'bold', color = clr)
ax0.set_xlabel("Time (min)")
ax0.set_ylabel("Current ($\mu$A)")
# Second axes
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(time, weight)
ax1.set_title("PSAL")
ax1.set_xlabel("Time (min)")
ax1.set_ylabel("Current ($\mu$A)")
# Third axes
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(time, weight)
ax2.axes.xaxis.set_visible(False)
ax2.axes.yaxis.set_visible(False)
# fourth axes
ax4 = fig.add_subplot(gs[2, 1])
ax4.set_axis_off()
ax4.text(0.05,0.5, summary, fontweight = 'bold')

ax3 = fig.add_subplot(gs[-1, :])
bbox = ax4.get_position()
rect = Rectangle((0,bbox.y0),1,bbox.height, color='gray', zorder=-1, transform=fig.transFigure, clip_on=False)
ax3.set_axis_off()

plt.tight_layout()
plt.show()

try:
    plt.plot([0, 1], [0,1,2])
except ValueError:
    plt.title('MISSING VALUE(s)')
