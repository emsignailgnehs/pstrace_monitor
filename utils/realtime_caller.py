import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
import warnings
from functools import partial



def extract_data(file):
    dataSource = ViewerDataSource()
    pickleFiles = [file]
    dataSource.load_picklefiles(pickleFiles)


    X, y, names,devices = removeDuplicates(*dataSource.exportXy())
    return X, y, names, devices


def get_range(array, low, high):
    # Return idxs and array values where array values are
    # between low and high
    array = np.array(array)
    idxs = np.where(
                np.logical_and(array >= low,
                               array <= high)
                )[0]
    
    return idxs, array[idxs]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]




class RealTimeCaller:
    
    def __init__(self, t0, pc0, dataln=None, smoothln=None, 
                 derivln=None, peakln=None, baseln=None, window=11):
        
        # Initialize variables
        self.i          = -1      # Index
        self.t          = []   # Time
        self.pc         = []  # Peak Current
        self.dy         = []    # 1st derivative
        self.smoothed   = []  # Smoothed peak currents
        
        self.window     = window    # Hanning func smoothing window
        self.normalizeRange = (2,3) # time bounds for normalization
        self.norm_val       = 1     # Normalization constant 
        self.norm_pc        = [] # Normalized currents
        self.threshold      = [0]   # Threshold line
        
        self.Ct             = 1000
        self.result         = False 
        self.flag           = False # Flag if positive result has been called
        self.call_time      = 0
        self.call_Sd        = 0
        
        # Line2D objects for real-time plotting
        self.dataln     = dataln
        self.smoothln   = smoothln
        self.derivln    = derivln
        self.peakln     = peakln
        self.baseln     = baseln
        self.lines = [self.dataln, self.smoothln, self.derivln,
                      self.peakln, self.baseln]
        
        self.update(data=(t0, pc0))
        
        
    def update(self, data, plots=False):
        
        t, pc = data
        
        self.i += 1
        self.t.append(t)
        self.pc.append(pc)
        
        self.smooth()
        self.normalize()
        self.calc_dy()
        self.find_peaks()
        self.calc_Ct()
        self.evaluate_result()
        
        if (self.result and not self.flag):
            # print(f'Called positive @ t = {t:0.2f} min.')
            self.flag      = True
            self.call_time = t
            self.call_Sd   = self.Sd
                
        if plots:
            self.update_lines()
            return *self.lines,
        
        else:
            return self.result
    
    
    def update_lines(self):
        # Update plot if doing realtime plotting
        self.dataln.set_data(self.t, self.pc)
        self.smoothln.set_data(self.t, self.smoothed)
        avg = np.mean(self.smoothed)
        self.derivln.set_data(self.t, avg+10*np.array(self.dy))
        
        if hasattr(self, 'peak_idx'):
            self.peakln.set_xdata([self.peak_time, self.peak_time])
            dy = np.array(self.dy)
            self.peakln.set_ydata([avg+10*dy[self.peak_idx],
                                   avg+10*(dy[self.peak_idx]-self.peak_props['prom'])
                                  ])
        
        if len(self.threshold) != 1:
            self.baseln.set_data(self.t, self.norm_val*self.threshold)
        return *self.lines,
    
    
    def smooth(self):
        # Smooth raw peak current data using Hanning window
        if len(self.t) <= self.window:
            self.smoothed.append(self.pc[-1])
            return
        
        w = np.hanning(self.window)
        w = w/w.sum()
        
        # Somehow applies window over array...
        x = self.pc
        s = np.r_[x[self.window-1:0:-1],
                  x,
                  x[-2:-self.window-1:-1]
                  ]
        arr = np.convolve(w, s, mode='valid')
        arr = arr[self.window//2 : -(self.window//2)]
        
        self.smoothed = arr
        return self.smoothed
     
    
    def normalize(self, mode='mean'):
        i = self.i
        
        if self.t[i] <= self.normalizeRange[1]:
            # Haven't recorded enough data yet
            return
        
        if self.norm_val != 1.0:
            # Already normalized
            self.norm_pc  = self.smoothed/self.norm_val
            return
        
        
        func = getattr(np, mode) #np.mean or np.max
        
        # Get time slice
        idxs, _ = get_range(self.t, self.normalizeRange[0],
                                    self.normalizeRange[1])
        
        pc = np.array(self.pc)
        norm_val = func(pc[idxs])
        if norm_val == 0:
            norm_val = 1.0
        self.norm_val = norm_val
        self.norm_pc  = np.array(self.smoothed)/self.norm_val
        return self.norm_val
    
        
    def calc_dy(self):
        if (self.norm_val == 1.0 or self.i < self.window):
            # Don't calculate derivative until after 
            # normalization time
            self.dy.append(0)
            return
        
        window = self.window
        if self.i >= 40:
            window = 31
        
        dy = savgol_filter(self.norm_pc, window, polyorder=3,
                           deriv=1, mode='nearest')
        dy *= -100
        self.dy = dy
        return self.dy
    
    
    def find_peaks(self, relheightlimit=0.9, widthlimit=0.05):
        
        if len(self.t) < 3:
            return
        
        t  = np.array(self.t)
        dy = np.array(self.dy)
        
        heightlimit = np.quantile(np.absolute(dy[0:-1] - dy[1:]), relheightlimit)
        peaks, props = find_peaks(dy,prominence = heightlimit,
                                   width = len(dy)*widthlimit, rel_height = 0.5)
        
        normalizer = (t[-1] - t[0])/len(t)
        
        try:
            idx   = np.argmax(props['prominences'])
            peak_time  = t[peaks[idx]]
            peak_prom  = props['prominences'][idx]
            peak_width = props['widths'][idx]*normalizer
            left_ips   = props['left_ips'][idx]*normalizer + t[0]
            self.peak_idx   = peaks[idx]
            self.peak_time  = peak_time
            self.peak_props = {
                               'prom':peak_prom,
                               'width':peak_width,
                               'left_ips':left_ips,
                               }
        except ValueError: # No valid peaks found
            pass
        
        return
    
    def calc_Ct(self, offset=0.05, method='hyper'):
        if not hasattr(self, 'peak_idx'):
            return
        if (method == 'hyper' or method == 'linear'):
            self._calc_Ct(offset, method)
        elif (method == 'baseline'):
            self._calc_Ct_from_bline(offset, method)
    
    def _calc_Ct(self, offset = 0.05, method='hyper'):
        
        t = np.array(self.t)
        y = np.array(self.norm_pc)
        left_ips = self.peak_props['left_ips']
        
        # Determine baseline region
        idxs, _ = get_range(t, 5, left_ips)
        if len(idxs) < 5: return
        lb, rb = idxs[0], idxs[-1]
        
        # Fitting functions
        def linear(t, p0, p1):
            return p0*t + p1
        
        def hyper(t, p0, p1, p2):
            return p0/(t+p1) + p2
                
        # Do fit
        func = {'hyper':hyper, 'linear':linear}.get(method)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            try:
                popt, pcov = curve_fit(func, t[lb:rb], y[lb:rb])
                self.threshold = func(t, *popt[:-1], (1-offset)*popt[-1])
            except:
                # Fit failed, use flat baseline
                # self.threshold = np.ones(len(t)) * (1 - offset)
                self.threshold = np.ones(len(t)) * (1 - offset) * np.mean(y[lb:rb])

        
        Ct_idx = 0
        for i, val in enumerate(y):
            if (t[i] > left_ips and val < self.threshold[i]):
                tval = self.threshold[i]
                Ct_idx = i
                break
                
        # Refine threshold crossing time by linear interpolation
        if Ct_idx != 0:
            # Must have crossed to the left of Ct_idx
            test_ts = np.linspace(t[Ct_idx-1], t[Ct_idx], 100)
            test_ys = np.linspace(y[Ct_idx-1], y[Ct_idx], 100)    
            nearest_idx = np.abs(test_ys - tval).argmin()
            self.Ct = test_ts[nearest_idx]
            
        return
    
    
    def _calc_Ct_from_bline(self, offset=0.05, method='baseline'):

        t = np.array(self.t)
        y = np.array(self.norm_pc)
        left_ips = self.peak_props['left_ips']
        
        # Determine baseline region
        idxs, _ = get_range(t, 1, left_ips)
        if len(idxs) < 5: return
        lb, rb = idxs[0], idxs[-1]
        
        # Find flattest 3-minute baseline
        def linear(t, m, b):
            return m*t + b
        
        if len(idxs) < 8:
            best, _ = curve_fit(linear, t[lb:rb], y[lb:rb])
        
        elif len(idxs) >= 8:
            best = [1e10, 1e10]
            lims = [(lb + i, lb + i + 8) for i in range(len(idxs)-8)]
            
            for (a, b) in lims:
                popt, pcov = curve_fit(linear, t[a:b], y[a:b])
                if abs(popt[0]) < abs(best[0]):
                    best = popt
        
        self.threshold = linear(t, *best)
        
        Ct_idx = 0
        for i, val in enumerate(y):
            if (t[i] > left_ips and val < self.threshold[i]):
                tval = self.threshold[i]
                Ct_idx = i
                break
                
        # Refine threshold crossing time by linear interpolation
        if Ct_idx != 0:
            # Must have crossed to the left of Ct_idx
            test_ts = np.linspace(t[Ct_idx-1], t[Ct_idx], 100)
            test_ys = np.linspace(y[Ct_idx-1], y[Ct_idx], 100)    
            nearest_idx = np.abs(test_ys - tval).argmin()
            self.Ct = test_ts[nearest_idx]
            
        return
        
    
    
    def evaluate_result(self, Ct_thresh=20, Sd_thresh=0.10):
        
        if not hasattr(self, 'peak_idx'):
            return
        
        # Starting signal: normalized current at left ips
        idx, _ = find_nearest(self.t, self.peak_props['left_ips'])
        
        self.Sd = self.norm_pc[idx] - self.norm_pc[-1]
        # print(f'{self.t[-1]:0.2f}, {self.Ct:0.2f}, {self.Sd:0.2f}')     
        if (self.Ct <= Ct_thresh and self.Sd >= Sd_thresh):
            self.result = True
            if not self.flag:
                self.call_Sd = self.Sd
                print(f'Called positive. t={self.t[-1]:0.2f}, Ct={self.Ct:0.2f}, Sd={self.Sd:0.2f}')
        
        else:
            self.result = False
        return
    
    
    def make_plot(self, title=''):        
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_title(title)
        ax.set_ylim(0, 50)
        ax.set_xlim(-1.5, 32)
        self.dataln, = ax.plot([],[], 'o')
        self.smoothln, = ax.plot([],[], '--', color='k', lw=2)
        self.derivln, = ax.plot([],[], '--', color='orange', lw=2)
        self.peakln,  = ax.plot([],[], '-', color='blue', lw=2)
        self.baseln,  = ax.plot([], [], '--', color='k', lw=1.5)
        
        self.update_lines()
        
        plt.show()
        return fig
        
  

class RealTimeSDAvg:
    def __init__(self, t0, pc0, dataln=None, avgln=None, 
                 threshln = None, window=11):
        
        # Initialize variables
        self.i          = -1      # Index
        self.t          = []   # Time
        self.y          = []   # Peak Current
        self.signals    = []
        self.filtY      = []
        self.avgFilter  = []
        self.stdFilter  = []
        self.threshold  = [0]
        
        self.lag        = 15
        # self.threshold  = 2
        self.influence  = 0.8
        
        self.window     = window    # Hanning func smoothing window
        self.normalizeRange = (2,3) # time bounds for normalization
        self.norm_val       = 1     # Normalization constant 
        self.norm_pc        = []    # Normalized currents
        
        self.Ct             = 1000
        self.result         = False 
        self.flag           = False # Flag if positive result has been called
        self.crossed        = False # If dy has crossed the rolling threshold
        self.call_time      = 0
        self.call_Sd        = 0
        
        # Line2D objects for real-time plotting
        self.dataln     = dataln
        self.avgln      = avgln
        self.threshln   = threshln
        self.lines = [self.dataln, self.avgln, self.threshln]
        
        self.update(data=(t0, pc0))
        
        
    def update(self, data, plots=True):
        
        t, pc = data
        
        self.i += 1
        self.t.append(t)    
        self.y.append(pc)
        
        self.smooth()
        self.calc_dy()
        
        self.check_crossing()
        
        if self.crossed:
            self.evaluate_result()
        
        self.thresholding_algo()
        self.evaluate_result()
        
        # if (self.result and not self.flag):
        #     # print(f'Called positive @ t = {t:0.2f} min.')
        #     self.flag      = True
        #     self.call_time = t
        #     self.call_Sd   = self.Sd
                
        if plots:
            self.update_lines()
            return *self.lines,
        
        # else:
        #     return self.result
    
    
    def update_lines(self):
        # Update plot if doing realtime plotting
        self.dataln.set_data(self.t, self.y)
        self.avgln.set_data(self.t, np.array(self.dy))
        self.threshln.set_data(self.t, np.array(self.threshold[:-1]))
       
        return *self.lines, 
    
    
    def smooth(self):
        # Smooth raw peak current data using Hanning window
        if len(self.t) <= self.window:
            self.filtY.append(self.y[-1])
            return self.filtY
        
        w = np.hanning(self.window)
        w = w/w.sum()
        
        # Somehow applies window over array...
        x = self.y
        s = np.r_[x[self.window-1:0:-1],
                  x,
                  x[-2:-self.window-1:-1]
                  ]
        arr = np.convolve(w, s, mode='valid')
        arr = arr[self.window//2 : -(self.window//2)]
        
        self.filtY = arr
        return self.filtY
    
    
    def calc_dy(self):
        
        if len(self.filtY) <= 1:
            self.dy = self.filtY
        
        else:
            self.dy = [self.filtY[i] - self.filtY[i-1] for
                       i in range(1, len(self.filtY))]
            self.dy = [self.dy[0]] + self.dy # Copy first value
            self.dy = -20*np.array(self.dy)
            self.dy = self.dy.tolist()
        
        self.dy = self.filtY
        
        return self.dy
    
    
    def check_crossing(self):
        # Check if the newest point crossed the threshold
        i = len(self.dy) - 1
        
        if i < (self.window + 5):
            return
        
        if (self.dy[i] < self.threshold[i]):
            if not self.crossed:
                self.crossed = True
    
    
    
    def thresholding_algo(self, factor=5, n=7):
        
        self.threshold += [0]
        i = len(self.dy) - 1
        
        if i <= self.window:
            return
        
        # Get slope of last n points
        idxs = np.arange(i-n, i+1)
        ddys  = [self.dy[j] - self.dy[j-1] for j in idxs]
        slope = np.mean(ddys)
        
        # Calculate threshold of next point
        next_limit = self.dy[i] + slope*factor
        self.threshold[i+1] = next_limit
    
        return
    
    
    def find_Ct(self):
        
        try:
            idx = np.where(self.dy == np.max(self.dy))[0][0]
            return idx, self.t[idx]
        except:
            return None, None
        # crossings = []
        # above = False
        # for i, _ in enumerate(self.dy):
        #     if (self.dy[i] <= self.threshold[i]) and not above:
        #         continue
        #     elif (self.dy[i] > self.threshold[i] and 
        #           self.threshold[i] > 0) and not above:
        #         above = True
        #         crossings.append(i)
        #     elif (self.dy[i] > self.threshold[i]) and above:
        #         continue
        #     elif (self.dy[i] <= self.threshold[i]) and above:
        #         above = False
        
        # if len(crossings) == 0:
        #     return None, None
        
        # idx = max(crossings)
        # Ct = self.t[idx]
        # return idx, Ct
            
        
    def evaluate_result(self, Ct_thresh=20, Sd_thresh=0.10):
                
        idx, self.Ct = self.find_Ct()
        if not idx:
            return
        
        self.Sd = (self.y[idx] - self.y[-1])/self.y[idx]
        # print(f'{self.t[-1]:0.2f}, {self.Ct:0.2f}, {self.Sd:0.2f}')     
        if (self.Ct <= Ct_thresh and self.Sd >= Sd_thresh):
            self.result = True
            if not self.flag:
                self.flag = True
                self.call_Sd = self.Sd
                print(f'Called positive. t={self.t[-1]:0.2f}, Ct={self.Ct:0.2f}, Sd={self.Sd:0.2f}')
        else:
            self.result = False
            
        # print(f't={self.t[-1]:0.2f}, Ct={self.Ct:0.2f}, Sd={self.Sd:0.2f}')

        return
        




class CallerSimulator(RealTimeCaller):
    
    def __init__(self, X, y, name, device):
        
        self.X      = X
        self.y      = y
        self.name   = name
        self.device = device
        
        super().__init__(X[0][0], X[1][0])
        
    def run(self):
        
        ts = self.X[0][1:]
        pcs = self.X[1][1:]
        
        for (t, pc) in zip(ts, pcs):
            self.update((t,pc))
            
        




if __name__ == '__main__':
    file = r'C:/Users/Elmer Guzman/SynologyDrive/RnD/Projects/LAMP-Covid Sensor/Data Export/20221102/20221102NewLMNSwabTest.picklez'
    
    from _util import ViewerDataSource
    from calling_algorithm import removeDuplicates
    
    X, y, names, devices = extract_data(file)
    
    X = X[10]
    t, pc = X[0], X[1]
    
    datastream = [(t[i], pc[i]) for i in range(1, len(t))]
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_ylim(0, 50)
    ax.set_xlim(-1.5, 32)
    # dataln, = ax.plot([],[], 'o')
    # smoothln, = ax.plot([],[], '--', color='k', lw=2)
    # derivln, = ax.plot([],[], '--', color='orange', lw=2)
    # peakln,  = ax.plot([],[], '-', color='blue', lw=2)
    # baseln,  = ax.plot([], [], '--', color='k', lw=1.5)    
    
    # rtc = RealTimeCaller(datastream[0][0], datastream[0][1], 
    #                      dataln, smoothln, derivln, peakln, baseln)
    
    
    dataln, = ax.plot([], [], 'o')
    avgln, = ax.plot([], [], '--', color='k', lw=2)
    stdln1, = ax.plot([], [], '--', color='orange', lw=1.2)
    stdln2, = ax.plot([], [], '--', color='orange', lw=1.2)
    
    rtc = RealTimeSDAvg(datastream[0][0], datastream[0][1],
                        dataln, avgln, stdln1, stdln2)
    
    
    ani     = FuncAnimation(fig, partial(rtc.update, plots=True), 
                            interval=10, blit = True,
                            frames = datastream[1:], repeat=False)

    plt.show()   
    
    # rtc.make_plot()
        
        
        
        