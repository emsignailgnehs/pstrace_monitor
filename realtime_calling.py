import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from utils._util import ViewerDataSource
from utils.calling_algorithm import (
    removeDuplicates,
    )
from utils.realtime_caller import RealTimeCaller, CallerRunner
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D







# file = r'C:/Users/Elmer Guzman/Desktop/covid sensor data/20221109NBBufferLODwithLeebio.picklez'
# file = r'C:/Users/Elmer Guzman/Desktop/covid sensor data/202211164ChannelsPTC.picklez'
# file = r'C:/Users/Elmer Guzman/SynologyDrive/RnD/Projects/LAMP-Covid Sensor/Data Export/20221102/20221102NewLMNSwabTest.picklez'

files = ["C:/Users/Elmer Guzman/Desktop/covid sensor data/20221017NewlyReceivedLMNQC.picklez",
"C:/Users/Elmer Guzman/Desktop/covid sensor data/20221018NewlyReceivedLMNQC.picklez",
"C:/Users/Elmer Guzman/Desktop/covid sensor data/20221025NewlyReceivedLMNQC.picklez",
"C:/Users/Elmer Guzman/Desktop/covid sensor data/20221026NewlyReceivedLMNQC.picklez",
"C:/Users/Elmer Guzman/Desktop/covid sensor data/20221027NewlyReceivedLMNQC.picklez",
"C:/Users/Elmer Guzman/Desktop/covid sensor data/20221102NewLMNSwabTest.picklez"]




def run_with_plot(t, pc):     
    datastream = [(t[i], pc[i]) for i in range(1, len(t))]
            
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_ylim(0, 50)
    ax.set_xlim(-1.5, 32)
    dataln, = ax.plot([],[], 'o')
    smoothln, = ax.plot([],[], '--', color='k', lw=2)
    derivln, = ax.plot([],[], '--', color='orange', lw=2)
    peakln,  = ax.plot([],[], '-', color='blue', lw=2)
    
    rtc = RealTimeCaller(t[0], pc[0], dataln, smoothln, derivln, peakln)
    ani     = FuncAnimation(fig, rtc.update_plot, interval=10, blit = True,
                            frames = datastream, repeat=False)
    plt.show()    


runners = []
for file in files:
    dataSource = ViewerDataSource()
    pickleFiles = [file]
    dataSource.load_picklefiles(pickleFiles)


    X, y, names,devices = removeDuplicates(*dataSource.exportXy())

    for i, (t, pc) in enumerate(X):
        runner = CallerRunner(t, pc, y[i])
        runners.append(runner)
        runner.run()


true_pos  = []
true_neg  = []
false_pos = []
false_neg = []
for r in runners:
    
    if r.res == '+':
        if r.y == '+':
            true_pos.append(r)
        elif r.y == '-':
            false_pos.append(r)
            
    if r.res == '-':
        if r.y == '+':
            false_neg.append(r)
        elif r.y == '-':
            true_neg.append(r)

total_pos = len(true_pos) + len(false_neg)
total_neg = len(true_neg) + len(false_pos)

ts = []
for r in true_pos:
    ts.append(r.call_time)

print('')
print(f'===== EVALUATED {len(runners)} FILES =====')
print(f'Found {len(true_pos)}/{total_pos} positives.')
print(f'Found {len(true_neg)}/{total_neg} negatives.')
print(f'{len(false_pos)} false positives.')
print(f'{len(false_neg)} false negatives.')
print('')
print(f'Evaluated true positives in {np.mean(ts):0.2f} +- {np.std(ts):0.2f} min')













