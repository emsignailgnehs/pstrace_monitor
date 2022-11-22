import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from utils._util import ViewerDataSource
from utils.calling_algorithm import (
    removeDuplicates, Pipeline, Smoother, Normalize, Truncate, Derivitive,
    FindPeak, HyperCt, CtPredictor
    )
from utils.realtime_caller import RealTimeCaller, CallerRunner
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D


def extract_data(file):
    dataSource = ViewerDataSource()
    pickleFiles = [file]
    dataSource.load_picklefiles(pickleFiles)


    X, y, names,devices = removeDuplicates(*dataSource.exportXy())
    return X, y, names, devices


def run_with_plot(file):  
    X, y, names, devices = extract_data(file)
    
    t, pc = X[2][0], X[2][1]
    
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
    return rtc, ani


def run_all(files):
    runners = []
    for file in files:
        X, y, names,devices = extract_data(file)
    
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

    return runners, true_pos, true_neg, false_pos, false_neg




class Comparator:
    
    def __init__(self, X, y, name, device):
        self.X = X
        self.y = y
        self.name = name
        self.device = device
        
    def standard_algo_call(self):
        hCtTPredictT = Pipeline([
            ('smooth', Smoother(stddev=2, windowlength=11, window='hanning')),
            ('normalize', Normalize(mode='mean', normalizeRange=(2,3))),
            ('truncate', Truncate(cutoffStart=2, cutoffEnd=25, n=90)),
            ('Derivitive', Derivitive(window=31, deg=3)),
            ('peak', FindPeak()),
            ('logCt',HyperCt()),
            ('predictor',CtPredictor(ct=20,prominence=0,sd=0.1))
        ])
        out = hCtTPredictT.transform([self.X])[0]
        prediction = {'res': bool(out[0]),
                          'Ct': out[1],
                          'Pr': out[2],
                          'Sd': out[3]}
        return prediction
    
    def realtime_algo_call(self):
        t, pc = self.X
        self.runner = CallerRunner(t, pc, self.y)
        runner = self.runner
        runner.run(printout=False)
        prediction = {'res': True if runner.res=='+' else False,
                      'Ct': runner.call_Ct,
                      'Pr': 0,
                      'Sd': runner.call_Sd}
        return prediction
    
    def compare(self):
        standard_pred = self.standard_algo_call()
        realtime_pred = self.realtime_algo_call()
        
        s = standard_pred['res']
        r = realtime_pred['res']

        d = {(1,1):'True positive',
             (1,0):'False positive',
             (0,1):'False negative',
             (0,0):'True negative'}       
        
        def _eval(res):
            return '+' if res else '-'
        
        if (d[(r, s)] == 'False positive' or 
            d[(r, s)] == 'False negative'):
            rtc = self.runner.rtc
            fig, ax = plt.subplots()
            ax.set_title(d[(r, s)])
            ax.plot(rtc.t, rtc.norm_pc)
        
        return _eval(r), _eval(s), d[(r, s)]




if __name__ == '__main__':
    file = r'C:/Users/Elmer Guzman/SynologyDrive/RnD/Projects/LAMP-Covid Sensor/Data Export/20221102/20221102NewLMNSwabTest.picklez'

    files = ["C:/Users/Elmer Guzman/Desktop/covid sensor data/20221017NewlyReceivedLMNQC.picklez",
    "C:/Users/Elmer Guzman/Desktop/covid sensor data/20221018NewlyReceivedLMNQC.picklez",
    "C:/Users/Elmer Guzman/Desktop/covid sensor data/20221025NewlyReceivedLMNQC.picklez",
    "C:/Users/Elmer Guzman/Desktop/covid sensor data/20221026NewlyReceivedLMNQC.picklez",
    "C:/Users/Elmer Guzman/Desktop/covid sensor data/20221027NewlyReceivedLMNQC.picklez",
    "C:/Users/Elmer Guzman/Desktop/covid sensor data/20221102NewLMNSwabTest.picklez"]
    
    # runners, true_pos, true_neg, false_pos, false_neg = run_all(files)
    
    # rtc, ani = run_with_plot(file)
    
    folder = r'C:\Users\Elmer Guzman\Desktop\covid sensor data'
    import os
    
    i = 0 # count total number of sensors evaluated
    d = {'True positive': [],
         'True negative': [],
         'False positive': [],
         'False negative': []}
    
    for file in os.listdir(folder):
        if file.endswith('.picklez'):
            f = os.path.join(folder, file)
            data = extract_data(f)
            
            data_list = [l for l in zip(*data)]
            for (X, y, name, device) in data_list:
                comp = Comparator(X, y, name, device)
                rt_res, st_red, res = comp.compare()
                
                d[res].append((file, name, device))
                print(f'{res}     {name}')
                
                i += 1
                
    
    
    








