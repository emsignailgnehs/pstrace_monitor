import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from utils._util import ViewerDataSource
from utils.calling_algorithm import (
    removeDuplicates, Pipeline, Smoother, Normalize, Truncate, Derivitive,
    FindPeak, HyperCt, CtPredictor
    )
from utils.realtime_caller import RealTimeCaller, CallerSimulator, RealTimeLinear
from matplotlib.animation import FuncAnimation
from functools import partial


def extract_data(file):
    dataSource = ViewerDataSource()
    pickleFiles = [file]
    dataSource.load_picklefiles(pickleFiles)
    # return dataSource


    X, y, names,devices = removeDuplicates(*dataSource.exportXy())
    return X, y, names, devices


def run_with_plot(file, i = 2):  
    X, y, names, devices = extract_data(file)
    
    X = X[i]
    t, pc = X[0], X[1]
    
    datastream = [(t[i], pc[i]) for i in range(0, len(t))]
            
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_ylim(0,50)
    ax.set_xlim(-1.5, 32)
    
    # dataln, = ax.plot([],[], 'o')
    # smoothln, = ax.plot([],[], '--', color='k', lw=2)
    # derivln, = ax.plot([],[], '--', color='orange', lw=2)
    # peakln,  = ax.plot([],[], '-', color='blue', lw=2)
    # baseln,  = ax.plot([], [], '--', color='k', lw=1.5)
    
    # rtc = RealTimeCaller(datastream[0][0], datastream[0][1],
    #                         dataln, smoothln, derivln, peakln, baseln)
    
    dataln, = ax.plot([], [], 'o')
    avgln, = ax.plot([], [], '--', color='k', lw=2)
    threshln, = ax.plot([], [], '--', color='orange', lw=1.2)
    
    rtc = RealTimeLinear(datastream[0][0], datastream[0][1],
                        dataln, avgln, threshln)
    
    
    ani     = FuncAnimation(fig, partial(rtc.update, plots=True), 
                            interval=50, blit = True,
                            frames = datastream[1:], repeat=False,
                            init_func = rtc.update_lines)
    
    # ani = CallerSimulator(X, y[i], names[i], devices[i])
    # ani.run()
    
    
    plt.show()   
     
    return rtc, ani


def run_all(files):
    runners = []
    for file in files:
        X, y, names,devices = extract_data(file)
    
        for i, _ in enumerate(X):
            runner = CallerSimulator(X[i], y[i], names[i], devices[i])
            runners.append(runner)
            runner.run()

    true_pos  = []
    true_neg  = []
    false_pos = []
    false_neg = []
    for r in runners:
        
        if r.result: #realtime predicts positive
            if r.y == '+':
                true_pos.append(r)
            elif r.y == '-':
                false_pos.append(r)
                
        if not r.result: # Realtime predicts negative
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
        self.runner = CallerSimulator(self.X, self.y, self.name, self.device)
        runner = self.runner
        runner.run()
        prediction = {'res': runner.result,
                      'Ct': runner.call_time,
                      'Pr': 0,
                      'Sd': runner.call_Sd,
                      'call_time': runner.call_time}
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
            rtc = self.runner
            # rtc.make_plot(title=d[(r, s)])
        
        return _eval(r), _eval(s), d[(r, s)], realtime_pred['call_time']


def compare_all(folder, n=20000):
    import os
    
    i = 0 # count total number of sensors evaluated
    d = {'True positive': [],
          'True negative': [],
          'False positive': [],
          'False negative': []}
    bads = []
    for file in os.listdir(folder):
        if file.endswith('.picklez'):
            f = os.path.join(folder, file)
            data = extract_data(f)
            
            data_list = [l for l in zip(*data)]
            for i, (X, y, name, device) in enumerate(data_list):
                comp = Comparator(X, y, name, device)
                rt_res, st_red, res, call_time = comp.compare()
                
                d[res].append((file, name, device, call_time))
                print(f'{res}     {file}')
                if res == 'False negative':
                    bads.append((f, i, comp))
                #     print(f'false negative {file}')
                if res == 'False positive':
                    bads.append((f, i, comp))
                #     print(f'false positive {file}')
                
                i += 1
            
            if i > n:
                break
                
    for key, l in d.items():
        print(f'{key}: {len(l)}')
        
    return d, bads, comp


if __name__ == '__main__':
    file = r'C:/Users/Elmer Guzman/SynologyDrive/RnD/Projects/LAMP-Covid Sensor/Data Export/20221102/20221102NewLMNSwabTest.picklez'

    # files = ["C:/Users/Elmer Guzman/Desktop/covid sensor data/20221017NewlyReceivedLMNQC.picklez",
    # "C:/Users/Elmer Guzman/Desktop/covid sensor data/20221018NewlyReceivedLMNQC.picklez",
    # "C:/Users/Elmer Guzman/Desktop/covid sensor data/20221025NewlyReceivedLMNQC.picklez",
    # "C:/Users/Elmer Guzman/Desktop/covid sensor data/20221026NewlyReceivedLMNQC.picklez",
    # "C:/Users/Elmer Guzman/Desktop/covid sensor data/20221027NewlyReceivedLMNQC.picklez",
    # "C:/Users/Elmer Guzman/Desktop/covid sensor data/20221102NewLMNSwabTest.picklez"
    # ]
    
    # runners, true_pos, true_neg, false_pos, false_neg = run_all(files)
    
    # ds = extract_data(file)
    # out_folder = r'C:\Users\Elmer Guzman\Desktop\simdata\C4'
    # import os
    # for i, (V, I) in enumerate(ds.rawView['data'][3]['data']['rawdata']):
    #     file = os.path.join(out_folder, f'{i}.csv')
    #     l = zip(V, I)
    #     with open(file, 'a') as f:
    #         for (V, I) in l:
    #             f.write(f'{V},{I}\n')
            
    
    rtc, ani = run_with_plot(file)
    
    # folder = r'C:\Users\Elmer Guzman\Desktop\covid sensor data'
    # d, bads, comp = compare_all(folder)
    
                
    
    
    








