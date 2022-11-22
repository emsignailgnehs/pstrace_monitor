import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
import warnings


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


class CallerRunner():
     # Class to simulate running calling algorithm in real time
    def __init__(self, t, pc, y):
        self.t = t
        self.pc = pc
        self.rtc = RealTimeCaller(t[0], pc[0])
        self.y = '+' if y else '-'
    
    def run(self, printout=True):
        frames = [(self.t[i], self.pc[i]) for i in range(1, len(self.t))]
        
        for (t, pc) in frames:
            self.rtc.update(t, pc)
            call_time, call_Ct, call_Sd = self.rtc.run_analysis()
            if (call_time != 0):
                self.call_time = call_time
                self.call_Ct   = call_Ct
                self.call_Sd   = call_Sd
                
                self.res = '+'
                break
        
        if self.rtc.result == False:
            self.res = '-'
            self.call_Ct = self.t[-1]
            self.call_Sd = 0
            self.call_time = self.t[-1]
        
        false_res = {('+', '-'):'FALSE POSITIVE',
                     ('-', '+'):'FALSE NEGATIVE',
                     ('-', '-'):'TRUE NEGATIVE',
                     ('+', '+'):'TRUE POSITIVE'}.get((self.res, self.y))
        
        if printout:
            print(f'P:{self.res}, M:{self.y}   {false_res}    t={self.call_Ct:0.1f}.')
        
            


class RealTimeCaller:
    
    def __init__(self, t, pc, dataln=None, smoothln=None, derivln=None, 
                 peakln=None, baseln=None, window=7):
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
        self.baseln  = baseln
        
        # Hardcoded constants
        self.normalizeRange = (2,3) # Normalize peak currents to average between these times
        self.norm_val = None
        self.norm_pc  = []
        self.Ct = None
        self.result = False
    
    def _slice(self, list, idxs):
        # list slicing by list of indices
        return [val for i,val in enumerate(list) if i in idxs]
    
        
    def update(self, t, pc):
        self.i += 1
        self.t.append(t)
        self.pc.append(pc)
        
        
        if bool(self.norm_val):
            # Already have norm. constant, just normalize the new value
            self.norm_pc.append(pc/self.norm_val)
            
        if (t > self.normalizeRange[1] and self.norm_val==None):
            # First time, do normalization
            vals = self.normalize(self.t, self.pc, mode='mean')
            self.norm_pc = vals
           
        
        self.smooth(self.i)
        self.update_dy()
    
    def run_analysis(self):        
        # Set data on dy line
        # Don't plot early dy/dt points
        # valid_idxs = self.truncate(5, 30)
        # t  = self._slice(self.t, valid_idxs)
        # dy = self._slice(self.dy, valid_idxs)
        # dy = np.array(dy)
        # self.derivln.set_data(t, 25+dy)
        # self.derivln.set_data(self.t[1:], self.dy)
        self.find_peaks()
        
        if hasattr(self, 'peak_props'):
            # Peak found
            self.estimate_ct()
        
        if (self.Ct and not self.result):
            pos = self.call_result()
            if pos:
                self.result = True
                return self.t[-1], self.Ct, self.Sd
        
        return 0,0,0
    
    
    def update_plot(self, data):
        # Get new data point
        t, pc = data
        self.update(t, pc)
        
        # Set data on raw and smoothed lines
        self.dataln.set_data(self.t, self.pc)
        self.smoothln.set_data(self.t, self.smoothed)
        
        # Set data on dy line
        # Don't plot early dy/dt points
        valid_idxs = self.truncate(5, 30)
        t  = self._slice(self.t, valid_idxs)
        dy = self._slice(self.dy, valid_idxs)
        dy = np.array(dy)
        self.derivln.set_data(t, 25+dy)
        # self.derivln.set_data(self.t[1:], self.dy)
        
        idx, peak_time, peak_prom = self.find_peaks()
        idx = np.where(self.t == peak_time)[0][0] if idx else 0
        self.peakln.set_xdata([peak_time, peak_time])
        self.peakln.set_ydata([25+self.dy[idx], 25+self.dy[idx] - peak_prom])
        # self.peakln.set_ydata([25, 30])
        
        if hasattr(self, 'peak_props'):
            # Peak found
            self.estimate_ct()
            self.baseln.set_data(self.t, self.threshold)
        
        if (self.Ct and not self.result):
            pos = self.call_result()
            if pos:
                self.result = True
                print(f'Called positive @ t = {self.t[-1]:0.2f}. Ct = {self.Ct:0.2f}, Sd = {self.Sd:0.2f}')
        
        return [self.dataln, self.smoothln, self.derivln, self.peakln, self.baseln]
        
        
    def smooth(self, i):                
        rbound = i
        lbound = i - self.window
        center = i - self.window//2
        
        if lbound < 0:
            self.smoothed.append(self.pc[i])
            self.dy.append(self.pc[i] - np.average(self.pc[0:i]))
            return self.pc[i]
        
        data = self.pc[lbound:rbound]
                
        # Hann window smoothing
        w = np.hanning(self.window)
        pt = np.convolve(w/w.sum(), data, mode='valid')[0]
        
        # Modify point and make the list longer
        self.smoothed[center] = pt
        self.smoothed.append(self.pc[i])
        
        
    def update_dy(self):
        if bool(self.norm_val):
            vals = self.norm_pc
        else:
            vals = self.smoothed
            
        dy = [vals[i] - vals[i-1] for i in range(1, len(self.t))]
        dy = [dy[0]] + dy
        try:
            dy = savgol_filter(dy, self.window, polyorder=1, 
                                    deriv=0)
            dy *= -100
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
        peaks, props = find_peaks(dy,prominence = heightlimit,
                                   width = len(dy)*0.05, rel_height = 0.5)
        # Choose most prominent peak
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
            return peaks[idx], peak_time, peak_prom,
        except:
            return 0,0,0
        
        
        
    
    def estimate_ct(self, offset=0.05):
        '''
        calculate Ct from threshold method
        
        threshold line is flat line "offset" below the normalization value
        '''
        t = np.array(self.t)
        pc = np.array(self.smoothed)
        left_ips = self.peak_props['left_ips']
        
        # self.threshold = (1-offset)*self.norm_val*np.ones(len(t))
        
        # Find region to fit between normalizeRange[1] and left_ips
        idxs = np.where(
                    np.logical_and(t > self.normalizeRange[0],
                                   t < left_ips - 1.5)
                    )[0]
        if len(idxs) < 5:
            self.threshold = np.zeros(len(t))
            return
        
        lb, rb = idxs[0], idxs[-1]
        
        def lin_eq(t, m, b):
            return m*t + b
        
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            popt, pcov = curve_fit(lin_eq, t[lb:rb], pc[lb:rb])
        m, b = popt
        self.threshold = lin_eq(t, m, 0.95*b)
        
        Ct_idx = 0
        for i, val in enumerate(pc):
            if t[i] > self.normalizeRange[1]:
                # print(self.t[i])
                if val < self.threshold[i]:
                    tval = self.threshold[i]
                    Ct     = t[i]
                    Ct_idx = i
                    break
        
        # Refine threshold crossing time by linear interpolation
        if Ct_idx != 0:
            test_ts = np.linspace(t[Ct_idx-1], t[Ct_idx], 1000)
            test_ys = np.linspace(pc[Ct_idx-1], pc[Ct_idx], 1000)    
            nearest_idx = np.abs(test_ys - tval).argmin()
            nearest_t = test_ts[nearest_idx]
            self.Ct = nearest_t   
        return
    
    
    def normalize(self, t, pc, mode='mean'):  
        # Get normalization constant
        pc = np.array(pc)
        func = getattr(np, mode)
        idxs = np.where(
                    np.logical_and(np.array(t) > self.normalizeRange[0],
                                   np.array(t) < self.normalizeRange[1])
                    )[0]
        norm_val = max(func(pc[min(idxs):max(idxs)]) , 1e-3)
        self.norm_val = norm_val
        norm_pc  = pc/norm_val
        return list(norm_pc)
    
    def call_result(self):
        
        Ct = self.Ct
        
        # Calculate signal drop
        # Starting signal = normalized pc at left ips
        start_idx, start_t = find_nearest(self.t, 
                                          self.peak_props['left_ips'])
        end_idx, end_t = find_nearest(self.t, 
                                      self.peak_props['left_ips'] + 5)
        
        start_pc = self.norm_pc[start_idx]
        end_pc   = self.norm_pc[end_idx]
        Sd       = start_pc - end_pc
        
        self.Sd = Sd
        
        if (Ct <= 20 and Sd >= 0.1):
            return 1
        else:
            return 0


