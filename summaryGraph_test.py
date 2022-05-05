import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams['figure.dpi'] = 150

plot_aspect = [4, 3] #[width, height]
width_ratios = [plot_aspect[0], plot_aspect[0], plot_aspect[0], plot_aspect[0]]

msg = ' | {"name": "20220411 N6O4/R18S 40 copy BioIVT (SL 1-3)", "deviceId": "AMS-ADM", "deviceSerial": "000005800170", "deviceDataId": null, "chipType": 4, "created": "2022-03-25T21:19:28.740427", "device": "AMS-ADM", "exp": "", "desc": ""} | {"stat": "ok", "avgTemp": "67.53 C", "chipInsertion": "29 / 29", "fluid": "C1-14/14, C4-14/14", "channelResult": "C1-Negative Ct:30.0 Pr:0.1 Sd:0.02,C4-Positive Ct:21.0 Pr:0.6 Sd:0.14"} | "Negative"'


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

for i in [0,1]:
    fig = plt.figure(figsize = (16, 5.5))
    gs = GridSpec(nrows=3, ncols=4, width_ratios=[1, 1, 1, 1], height_ratios=[3, 2, 0.1])

    time = [0, 1]
    height = [0, 1]
    weight = [0, 1]
    # First axes
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(time, height)
    ax0.axes.xaxis.set_visible(False)
    ax0.axes.yaxis.set_visible(False)
    # Second axes
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(time, weight)
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    # Third axes
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(time, weight)
    ax2.axes.xaxis.set_visible(False)
    ax2.axes.yaxis.set_visible(False)
    # fourth axes
    ax3 = fig.add_subplot(gs[-1, :])
    ax3.set_axis_off()
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axes.xaxis.set_visible(False)
    ax4.axes.yaxis.set_visible(False)
    ax4.plot(time, weight)
    # ax3.plot(time, height)
    # text = "123456789"

    # ax3.text(0.5,0.5, "text")
plt.tight_layout()
plt.show()
