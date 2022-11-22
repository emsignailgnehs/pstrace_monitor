import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from utils._util import ViewerDataSource
from utils.calling_algorithm import (
    removeDuplicates,
    )
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D






# file = r'C:/Users/Elmer Guzman/Desktop/covid sensor data/20221109NBBufferLODwithLeebio.picklez'
# file = r'C:/Users/Elmer Guzman/Desktop/covid sensor data/202211164ChannelsPTC.picklez'
file = r'C:/Users/Elmer Guzman/SynologyDrive/RnD/Projects/LAMP-Covid Sensor/Data Export/20221102/20221102NewLMNSwabTest.picklez'

dataSource = ViewerDataSource()
pickleFiles = [file]
dataSource.load_picklefiles(pickleFiles)


X, y, names,devices = removeDuplicates(*dataSource.exportXy())


t  = X[2][0] # time
pc = X[2][1] # peak current






class AnimatedPlotter:
    def __init__(self, ax, dataline, smoothline):
        self.ax = ax
        self.tdata = []
        self.ydata = []
        self.smoothdata = []
        self.dataline = dataline
        self.smoothline = smoothline
        self.ax.set_ylim(16, 42)
        self.ax.set_xlim(-1.5, 32)

    def update(self, data):
        
        self.tdata.append(data[0])
        self.ydata.append(data[1])
        
        self.dataline.set_data(self.tdata, self.ydata)
        return self.dataline,




class RealTimeCaller:
    
    def __init__(self, t, pc, dataln, smoothln, derivln, peakln,
                 window=7):
        self.i  = 0
        self.t  = [t]
        self.pc = [pc]
        self.window = window
        self.smoothed = [pc]
        self.dy = [0]
        
        self.dataln = dataln
        self.smoothln = smoothln
        self.derivln = derivln
        self.peakln  = peakln
    
    def _slice(self, list, idxs):
        # list slicing by list of indices
        return [val for i,val in enumerate(list) if i in idxs]
    
    def update(self, t, pc):
        self.i += 1
        self.t.append(t)
        self.pc.append(pc)
        self.smooth(self.i)
        self.update_dy()
    
    def update_plot(self, data):
        t, pc = data
        self.update(t, pc)
        
        self.dataln.set_data(self.t, self.pc)
        self.smoothln.set_data(self.t, self.smoothed)
        
        # Don't plot early dy/dt points
        valid_idxs = self.truncate(5, 30)
        t  = self._slice(self.t, valid_idxs)
        dy = self._slice(self.dy, valid_idxs)
        self.derivln.set_data(t, dy)
        # self.derivln.set_data(self.t[1:], self.dy)
        
        idx, peak_time, peak_prom = self.find_peaks()
        idx = np.where(self.t == peak_time)[0][0] if idx else 0
        self.peakln.set_xdata([peak_time, peak_time])
        self.peakln.set_ydata([self.dy[idx], self.dy[idx] - peak_prom])
        # self.peakln.set_ydata([25, 30])
        
        return [self.dataln, self.smoothln, self.derivln, self.peakln]
        
        
    def smooth(self, i):                
        rbound = i
        lbound = i - self.window
        center = i - self.window//2
        
        if lbound < 0:
            self.smoothed.append(self.pc[i])
            self.dy.append(self.pc[i] - np.average(self.pc[0:i]))
            return self.pc[i]
        
        data = self.pc[lbound:rbound]
        
        # Savitzky-Golay filter
        # smoothed_pts = savgol_filter(data, self.window, polyorder=1,
        #                                   deriv=1)
        
        # pt = smoothed_pts[self.window//2]
        # pt *= -10
        # pt += 25
        
        # Hann window smoothing
        w = np.hanning(self.window)
        pt = np.convolve(w/w.sum(), data, mode='valid')[0]
        
        # Modify point and make the list longer
        self.smoothed[center] = pt
        self.smoothed.append(self.pc[i])
        
        
    def update_dy(self):
        dy = [self.smoothed[i] - self.smoothed[i-1]
                   for i in range(1, len(self.t))]
        dy = [dy[0]] + dy
        try:
            dy = savgol_filter(dy, self.window, polyorder=1, 
                                    deriv=0)
            dy *= -10
            dy += 25
        except:
            # Not enough data to fit filter yet
            pass
        self.dy = dy
        return self.dy
    
    
    def truncate(self, cutoffStart=4, cutoffEnd=25):
        # Return indices that fall between cutoffStart and cutoffEnd
        idxs = [i for i,t in enumerate(self.t) 
                if (t >= cutoffStart and t <= cutoffEnd)]
        return idxs
    
    
    def find_peaks(self):
        valid_idxs = self.truncate(2, 30)
        if len(valid_idxs) < 10:
            return 0,0,0
        t  = np.array(self._slice(self.t, valid_idxs))
        dy = np.array(self._slice(self.dy, valid_idxs))
        
        heightlimit = np.quantile(np.absolute(dy[0:-1] - dy[1:]), 0.3)
        # peaks, props = find_peaks(dy,prominence = heightlimit,
                                  # width = len(dy)*0.05, rel_height = 0.5)
        peaks, props = find_peaks(dy, prominence = heightlimit)
        # Choose most prominent peak
        normalizer = (t[-1] - t[0])/len(t)
        try:
            idx   = np.argmax(props['prominences'])
            peak_time  = t[peaks[idx]]
            peak_prom  = props['prominences'][idx]
            # peak_width = props['widths'][idx]*normalizer
            # left_ips   = props['left_ips'][idx]*normalizer + t[0]
            return peaks[idx], peak_time, peak_prom
        except:
            return 0,0,0
        
        
        
    
    def estimate_ct(self):
        return
    
    def normalize(self):
        return
    
    def call_result(self):
        return
        
            

# rtc = RealTimeCaller(t[0], pc[0])
# for i in range(1, len(t)):
#     rtc.update(t[i], pc[i])


# fig, ax = plt.subplots(figsize=(10,10))
# ax.plot(t, pc, 'o')
# ax.plot(t, np.array(rtc.smoothed), '-')

datastream = [(t[i], pc[i]) for i in range(1, len(t))]
        
fig, ax = plt.subplots(figsize=(10,10))
ax.set_ylim(0, 50)
ax.set_xlim(-1.5, 32)
dataln, = ax.plot([],[], 'o')
smoothln, = ax.plot([],[], '--', color='k', lw=2)
derivln, = ax.plot([],[], '--', color='orange', lw=2)
peakln,  = ax.plot([],[], '-', color='blue', lw=2)

rtc = RealTimeCaller(t[0], pc[0], dataln, smoothln, derivln, peakln)
ani     = FuncAnimation(fig, rtc.update_plot, interval=50, blit = True,
                        frames = datastream, repeat=False)
plt.show()    
