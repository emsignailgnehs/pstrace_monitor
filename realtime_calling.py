import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
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
import re


def remove_saliva_runs(X, y, names, devices):
    new_X = []
    new_y = []
    new_names = []
    new_dev = []
    for (xi, yi, name, device) in zip(X, y, names, devices):
        if "Saliva" in name.split(' '):
            continue
        new_X.append(xi)
        new_y.append(yi)
        new_names.append(name)
        new_dev.append(device)
    return new_X, new_y, new_names, new_dev


def extract_data(file):
    dataSource = ViewerDataSource()
    pickleFiles = [file]
    dataSource.load_picklefiles(pickleFiles)
    
    
    # Parse filename to get experiment conditions
    
    mxs = {'1NM': '1X NM',
           '01NM': '0.1X NM'}    
    
    l = file.split('-')
    matrix = l[1]
    matrix = mxs.get(matrix, matrix)
    cp = l[2].replace('.picklez', '')
    
    X, y, names,devices = removeDuplicates(*dataSource.exportXy())

    matrices       = []
    concentrations = []    
        
    
    # Get virus loading from name
    for n, name in enumerate(names):
        
        matrices.append(matrix)
        concentrations.append(cp)
    
    for i, c in enumerate(concentrations):
        if (c == 0) or (c == 'NTC'):
            y[i] = 0
        else:
            y[i] = 1

    return X, y, names, concentrations, matrices, devices



def run_with_plot(file, i = 13):  
    X, y, names, concs, devices = extract_data(file)

    X = X[i]
    t, pc = X[0], X[1]
    
    datastream = [(t[i], pc[i]) for i in range(0, len(t))]
            
    fig, ax = plt.subplots(figsize=(5,5), dpi=100)
    ax.set_ylim(0, 2)
    ax.set_xlim(-1.5, 32)  
    ax.set_xlabel('Time/ min')
    ax.set_ylabel('Normalized Current')
    dataln, = ax.plot([], [], 'o')
    avgln, = ax.plot([], [], '--', color='k', lw=2)
    threshln, = ax.plot([], [], '--', color='orange', lw=1.2)
    # import time
    # time.sleep(1)
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




def make_truth_table(earlycalls, ground_truth):
    def comp(val, truth, category):
        if category == 'true pos':
            if val == truth == 1:
                return 1
        if category == 'true neg':
            if val == truth == 0:
                return 1
        if category == 'false pos':
            if val == 1 and truth == 0:
                return 1
        if category == 'false neg':
            if val == 0 and truth == 1:
                return 1
        return 0
    
    calls = []
    for key in sorted(earlycalls.keys()):
        calls.append(earlycalls[key])
    cats = ['true pos', 'true neg', 'false pos', 'false neg']
    truth_table = []
    for cat in cats:
        l = []
        for val in calls:
            l.append(comp(val, ground_truth, cat))
        truth_table.append(np.array(l))
    return np.array(truth_table)


class Comparator:
    # compare realtime algorithm to full time algorithm
    def __init__(self, X, y, name, conc, preheat_time, device, plot, kwargs):
        # kwargs to pass to RealTimeCaller class
        self.X = X
        self.y = y
        self.name = name
        self.conc = conc
        self.preheat_time = preheat_time
        self.device = device
        self.plot   = plot
        self.kwargs = kwargs
        
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
        self.runner = CallerSimulator(self.X, self.y, self.name, 
                                      self.device, self.kwargs)
        runner = self.runner
        runner.run()
        prediction = {'res': runner.result,
                      'Ct': runner.Ct,
                      'Pr': 0,
                      'Sd': runner.call_Sd,
                      'call_time': runner.call_time,
                      'earlycalls': runner.earlycalls}
        return prediction
    
    def compare(self):
        self.standard_pred = self.standard_algo_call()
        self.realtime_pred = self.realtime_algo_call()
        
        # s = self.standard_pred['res']
        s = self.y
        r = self.realtime_pred['res']

        d = {(1,1):'True positive',
             (1,0):'False positive',
             (0,1):'False negative',
             (0,0):'True negative'}       
        
        def _eval(res):
            return '+' if res else '-'
        
        if (d[(r, s)] == 'False positive' or 
            d[(r, s)] == 'False negative'):
            rtc = self.runner
            if self.plot:
                rtc.make_plot(title=f'{self.name} {d[(r, s)]}',
                              text=f'{self.conc} cp')
        
        # make truth table
        # ground truth = base algorithm
        # tt = make_truth_table(self.realtime_pred['earlycalls'], s)
        
        # ground truth = user mark
        tt = make_truth_table(self.realtime_pred['earlycalls'], self.y)
        
        return d[(r, s)], tt, self.realtime_pred['call_time']



def compare_all(folder, n=20000, plot_mismatch=False, kwargs={}):
    import os
    
    i = 0 # count total number of sensors evaluated
    d = {'True positive': [],
          'True negative': [],
          'False positive': [],
          'False negative': []}
    bads = []
    summed_table = None
    import glob
    # for file in os.listdir(folder):
    j = 0
    for file in glob.glob(folder + '/**', recursive=True):
        if file.endswith('.picklez'):
            f = os.path.join(folder, file)
            data = extract_data(f)
            X, y, names, concs, preheat_times, devices = data
            
            data_list = [l for l in zip(*data)]
            for i, (X, y, name, conc, preheat_time, 
                    device) in enumerate(data_list):
                
                # Filter to only analyze certain data
                # if preheat_time != 5:
                #     continue
                # if conc != 10000:
                #     continue
                
                # print(y, conc, name)
                
                if j > n: break
                comp = Comparator(X, y, name, conc, preheat_time, device, 
                                  plot_mismatch, kwargs)
                res, truth_table, call_time = comp.compare()
                if summed_table is None:
                    # Count # of true pos, true neg, false pos, false neg
                    summed_table = truth_table
                else:
                    summed_table += truth_table
                
                d[res].append((file, name, conc, device, call_time, comp))
                # print(f'{res}     {file}')
                call, sd, ct =  (comp.runner.call_time, 
                                 comp.runner.Sd, 
                                 comp.runner.Ct)
                if res == 'False negative':
                    bads.append((f, i, comp))
                    # print('')
                    # print(f'false negative {file}')
                    print(f'false negative {comp.conc}')
                    # print(f'Call time {call:0.2f}, Sd: {sd:0.2f}, Ct:{ct:0.2f}')                 
                if res == 'False positive':
                    bads.append((f, i, comp))
                    # print('')
                    # print(f'false positive {file}')
                    print(f'false positive {comp.conc}')
                    # print(f'Call time {call:0.2f}, Sd: {sd:0.2f}, Ct:{ct:0.2f}')                 
                j += 1
                
            
                
    # for key, l in d.items():
    #     print(f'{key}: {len(l)}')
        
    _, _, _, _, calltime, _ = zip(*d['True positive'])
    print(f'Call time (true positive):{np.mean(calltime):0.2f} +- {np.std(calltime):0.2f} min')
    # print(f'Median   :{np.median(calltime):0.2f} +- {np.quantile(calltime, 0.75):0.2f} min')
    
    return d, bads, comp, summed_table




def calltime_histogram(d, title, name):
    fig, ax = plt.subplots(figsize=(5,5), dpi=150)
    bins = np.arange(7, 30, 1)
    last_heights = np.array([0 for _ in bins][:-1])
    
    true_pos  = []
    false_pos = []
    
    for _,_,_,_,_, comp in d['True positive']:
        true_pos.append( (comp.runner.call_time,
                          comp.preheat_time) )
        
    for _,_,_,_,_, comp in d['False positive']:
        false_pos.append(comp.runner.call_time)   
    
    # Color by concentration
    concs = set([conc for _, conc in true_pos if conc != 0])
    concs = sorted(list(concs))  
        
    def color(cp):
        # maximum = 5
        # x = 0.3 + 0.7*float(cp)/maximum
        # arr = plt.cm.Greens(np.linspace(x,x,1))
        # return arr[0]
    
        if cp == '0.1X NM':
            return 'limegreen'
        if cp == '1X NM':
            return 'darkgreen'
        if cp == 'NTC':
            return 'blue'
        
        
    
    for i, conc in enumerate(concs):
        l = [ct for (ct, c) in true_pos if c == conc]
        if len(l) == 0: continue
        # Make histogram of just this concentraiton
        titlestring = f'N={len(l)}, {np.mean(l):0.2f} +- {np.std(l):0.2f} min'
        subfig, subax = plt.subplots(figsize=(5,5), dpi=150)
        subax.hist(l, bins=bins, rwidth=0.9, color=color(conc),
                   label=f'{conc}')
        subax.set_xlabel('Call time/ min')
        subax.set_ylabel('Count')
        subax.set_title(titlestring)
        subax.legend()
        
        # Plot to overlay histogram
        heights, locs = np.histogram(l, bins=bins)
        locs = locs[:-1]
        ax.bar(locs, heights, bottom=last_heights, color=color(conc),
               label=f'{conc} min')
        last_heights += heights
        
        
    
    # Plot false negatives to overlay
    heights, locs = np.histogram(false_pos, bins=bins)
    locs = locs[:-1]
    ax.bar(locs, heights, bottom = last_heights, color='red')
    
    ax.set_xlabel('Call time/ min')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    # plt.show()
    # fname = os.path.join(r'C:\SynologyDrive\Brian\Calling algorithm\realtime calling data\varying SD histograms', 
    #                      name +'.png')
    # plt.savefig(fname)
    # plt.close()


if __name__ == '__main__':    
    # file = r'C:/Users/Elmer Guzman/Desktop/covid sensor data/20221103Test.picklez'
    # rtc, ani = run_with_plot(file, 10)
    
    # file = r'C:/SynologyDrive/Brian/Calling algorithm/New collected data/20230131 2000cpswab 5min R18S C4 Run1.picklez'
    # file = r'C:/SynologyDrive/Brian/Calling algorithm/New collected data/20230131 NTC 5min N6O4 C1 Run1.picklez'
    # rtc, ani = run_with_plot(file, 3)
    
    # Old data:
    # folder = r'C:\Users\Elmer Guzman\Desktop\covid sensor data'
    
    # New data:
    folder = r'C:\SynologyDrive\Brian\Calling algorithm\Nasal matrix data'
    for thresh in [0.1, 0.08, 0.06]:
    # for thresh in [0.1]:
        print(f' ==== Threshold = {thresh} ====')
        print('')
        d, bads, comp, tt = compare_all(folder,
                                    plot_mismatch=True,
                                    kwargs={'Sd_thresh': thresh})
        df = pd.DataFrame(tt, index=['True positive', 'True negative', 'False positive', 'False negative'],
                          columns = ['17.5', '20.0', '22.5', '25.0', '27.5', '30.0'])
        print('')
        print(df)
        calltime_histogram(d, f'', name=str(int(100*thresh)))
        print('')
        print('')
        
    # compare_params(folder)
    


    
                
    
    
    








