import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import precision_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import joblib,os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.metrics import precision_score, recall_score
from scipy.signal import savgol_filter
from scipy import signal
from scipy.optimize import least_squares

"""
Update Note:
2022/11/17: Change the Ct definition to Ct = left_ips
"""

_version = '2stage_prescan_v1.0.8'

def get_picklez(folder):
    fs = []
    for root,fds,files in os.walk(folder):        
        for f in files:
            if f.endswith('picklez'):
                fs.append(os.path.join(root,f))
    return fs
    
def removeDuplicates(*args):    
    currents = set()
    ids = []
    for t,c in args[0]:
        if sum(c) in currents:
            ids.append(False)
        else:
            ids.append(True)
            currents.add(sum(c))
    return [i[ids] for i in args]
        

def findTimeVal(t,val,t0,dt):
    t0idx = int((t0 - t[0]) / (t[-1]-t[0]) * len(val))
    t1idx = int((t0 +dt - t[0]) / (t[-1]-t[0]) * len(val))
    return val[max(0,t0idx):t1idx]

 
def convert_list_to_X(data):
    """
    data is the format of:
    [[ [t1,t2...],[c1,c2...]],...]
    convert to numpy arry, retain the list of t1,t2... and c1,c2...
    """
    if not data: return np.array([])
    X = np.empty((len(data),2),dtype=list)
    X[:] = data
    return X


def getDataFromPicklez(*datalist,):
    """
    pull the data from picklez file. 
    return X and y.
    """    
    traces=[]
    userMark = []
    for data in datalist:
        ps = data['pstraces']
        for k,value in ps.items():        
            for d in value:
                mark = d.get('userMarkedAs',None)
                if mark:
                    t = timeseries_to_axis(d['data']['time'])
                    pc = [i['pc'] for i in d['data']['fit']]
                    traces.append((t,pc))
                    userMark.append(int(mark=='positive'))            
    return convert_list_to_X(traces),np.array(userMark)

def smooth(x,windowlenth=11,window='hanning'):
    "windowlenth need to be an odd number"
    s = np.r_[x[windowlenth-1:0:-1],x,x[-2:-windowlenth-1:-1]]
    w = getattr(np,window)(windowlenth)
    return np.convolve(w/w.sum(),s,mode='valid')[windowlenth//2:-(windowlenth//2)]
    
def timeseries_to_axis(timeseries):
    "convert datetime series to time series in minutes"
    return [(d-timeseries[0]).seconds/60 for d in timeseries]

def traceProcessPipe(
        outlier_para={"stddev":2,},
        smooth_para={"windowlenth":11,"window":'hanning'},
        extractTP_para={"cutoffStart":0,"cutoffEnd":60,"n":150}):
    """
    process the [time,pc] data and 
    return the processed value to be used for prediction
    """
    def pipe(row):
        t,pc = row
        t,pc = reject_outliers(t,pc,**outlier_para)
        pc = smooth(pc,**smooth_para)
        row = extract_timepionts(t,pc,**extractTP_para)
        row = normalize(row)
        return row
    return pipe

def normalize(row):
    return row/np.max(row)

def reject_outliers(time,data, stddev=2):
    '''remove the outlier from time series data, 
    stddev is the number of stddev for rejection.
    '''
    sli = abs(data - np.mean(data)) <= stddev * np.std(data)
    return np.array(time)[sli], np.array(data)[sli]



def extract_timepionts(time,data,cutoffStart, cutoffEnd=60,n=150):
    '''
    extract time and data with time<=cutoff
    n points of data will be returned.
    unknown values are from interpolation. 
    '''
    tp = 0
    datalength = len(data)
    newdata = []
    endslope = np.polyfit(time[-11:],data[-11:],deg=1)
    for i in np.linspace(cutoffStart,cutoffEnd,n):
        for t in time[tp:]:
            if t>=i:
                break
            tp+=1
        if tp+1>= datalength:
            #if the new timepoint is outside of known range, use last 11 points to fit the curve.
            newdata.append(i*endslope[0]+endslope[1])
        else:
            #otherwise interpolate the data.
            x1 = time[tp]
            y1 = data[tp]
            x2 = time[tp+1]
            y2 = data[tp+1]        
            newdata.append( y1 + (i-x1)*(y2-y1)/(x2-x1) )
    return np.array(newdata)
    
    
    
class SmoothTruncateNormalize(BaseEstimator,TransformerMixin):
    def __init__(self,**params):
        self.params=params        
    def fit(self,X,y=None):
        return self 
    def transform(self,X,y=None):
        return np.apply_along_axis(traceProcessPipe(**self.params),1,X,)

class Smoother(BaseEstimator,TransformerMixin):
    def __init__(self,stddev=2,windowlength=11,window='hanning'):
        self.stddev = stddev
        self.windowlength = windowlength
        self.window = window
    def fit(self,X,y=None):
        return self 
    def transformer(self,X):
        t,pc = X
        t,pc = reject_outliers(t,pc,stddev=self.stddev)
        pc = smooth(pc,windowlenth=self.windowlength,window=self.window)
        return [t,pc]
    def transform(self,X,y=None):        
        # return np.apply_along_axis(self.transformer,1,X,)
        return np.array([self.transformer(i) for i in X],dtype='object')

 

class Derivitive(BaseEstimator,TransformerMixin):
    def __init__(self,window=31,deg=3):
        self.window = window
        self.deg = deg
        
        
    def fit(self,X,y=None):
        return self 
    def transformer(self,X):
        t,pc = X
        ss = savgol_filter(pc,window_length=self.window,polyorder=self.deg,deriv=1)        
        return [t,-ss,pc]
    def transform(self,X,y=None):        
        # return np.apply_along_axis(self.transformer,1,X,)
        return np.array([self.transformer(i) for i in X],dtype='object')


class FindPeak(BaseEstimator,TransformerMixin):
    def __init__(self,heightlimit=0.9,widthlimit=0.05):
        self.heightlimit = heightlimit
        self.widthlimit = widthlimit
        
    def fit(self,X,y=None):
        return self 
    
    def transformer(self,X):
        
        t,gradient,pc = X
        heightlimit = np.quantile(np.absolute(gradient[0:-1] - gradient[1:]), self.heightlimit)
        peaks,props = signal.find_peaks(gradient,prominence=heightlimit,width= len(gradient) * self.widthlimit, rel_height=0.5)
        
        
        peak_pos,left_ips,peak_prominence,peak_width = (t[-1],t[-1],0,0)
        sdAtRightIps,sdAt3min,sdAt5min,sdAt10min,sdAt15min,sdAtEnd = (0,0,0,0,0,0)
        if len(peaks) != 0:            
        # most prominent peak in props 
            tspan = t[-1]-t[0]
            normalizer =  tspan / len(gradient) 
            maxpeak_index = props['prominences'].argmax()
            peak_pos = peaks[maxpeak_index] * normalizer + t[0]
            peak_prominence = props['prominences'][maxpeak_index] 
            peak_width = props['widths'][maxpeak_index] * normalizer 
            left_ips = props['left_ips'][maxpeak_index] * normalizer  + t[0]

            pcMaxIdx = len(pc) - 1

            # siganl at left_ips:
            startPosition = int(props['left_ips'][maxpeak_index])
            sStart = pc[startPosition]
            # find signal drop at different positions:
            # sigal drop at peak_width 
            sdAtRightIps = sStart - pc[min(int(props['right_ips'][maxpeak_index]), pcMaxIdx)]
            # signal drop at 3 min later
            sdAt3min = sStart - pc[min(startPosition + int(3 / normalizer), pcMaxIdx)]
            # signal drop at 5 min later
            sdAt5min = sStart - pc[min(startPosition + int(5 / normalizer), pcMaxIdx)]
            # signal drop at 10 min later
            sdAt10min = sStart - pc[min(startPosition + int(10 / normalizer), pcMaxIdx)]
            # siganl drop at 15 min later
            sdAt15min = sStart - pc[min(startPosition + int(15 / normalizer), pcMaxIdx)]
            # signal drop at end       
            sdAtEnd = sStart - pc[-1]            
            
            # calculate threshold Ct
            
        return [left_ips,peak_prominence*100,peak_width,sdAtRightIps,sdAt3min,sdAt5min,sdAt10min,sdAt15min,sdAtEnd,t,gradient,pc]
        
    def transform(self,X,y=None):        
        # return np.apply_along_axis(self.transformer,1,X,)
        return np.array([self.transformer(i) for i in X],dtype='object')

class ThresholdCt(BaseEstimator,TransformerMixin):
    "calculate the Ct from threshold method,this should always come after findPeak"
    def __init__(self,degree=1,fitwindow=4,offset=0.05):
        """
        degree is the polyfit degree of fitting,
        fitwindow is how many minutes aheat should the algorithm fit to
        offset is how much the fitted curve shifts down. this is in relative scale to the intial fitting point.
        """
        self.degree = degree
        self.fitwindow=fitwindow
        self.offset = offset
                
    def fit(self,X,y=None):        
        return self    
    def transformer(self,X):
        fitwindow = self.fitwindow
        degree = self.degree
        offset = self.offset
        t,deri,smoothed_c = X[-3:]
        left_ips,peak_prominence,peak_width = X[0:3]
        tofit = findTimeVal(t,smoothed_c,left_ips-fitwindow,fitwindow)
        fitpara = np.polyfit(np.linspace(max(left_ips-fitwindow,t[0]),left_ips,len(tofit)),np.array(tofit,dtype=float),deg=degree)        
        threshold = (tofit[-1]) * offset
        thresholdpara = fitpara + np.array( [0]*degree +[-threshold])
        thresholdline = np.poly1d(thresholdpara) 
        tosearch = findTimeVal(t,smoothed_c,left_ips,t[-1])
        tosearchT = np.linspace(left_ips,t[-1],len(tosearch))
        thresholdSearch = thresholdline(tosearchT) - tosearch
        thresholdCt = left_ips
        for sT,sthre in zip(tosearchT,thresholdSearch):        
            if sthre > 0:
                break
            thresholdCt = sT
        return  [*X[0:-3],*thresholdpara,thresholdCt]
          
    def transform(self,X,y=None):        
        return np.array([self.transformer(i) for i in X])
    
class LogCt(BaseEstimator,TransformerMixin):
    "calculate the Ct from threshold method,this should always come after findPeak"
    def __init__(self,degree=1,fitwindow=4,offset=0.05):
        """
        degree is the polyfit degree of fitting,
        fit on the log values of the curve
        fitwindow is how many minutes aheat should the algorithm fit to
        offset is how much the fitted curve shifts down. this is in relative scale to the intial fitting point.
        """
        self.degree = degree
        self.fitwindow=fitwindow
        self.offset = offset
                
    def fit(self,X,y=None):        
        return self    
    def transformer(self,X):
        fitwindow = self.fitwindow
        degree = self.degree
        offset = self.offset
        t,deri,smoothed_c = X[-3:]
        left_ips,peak_prominence,peak_width = X[0:3]
        _tofit = findTimeVal(t,smoothed_c,left_ips-fitwindow,fitwindow)
        _tofitmin = len(_tofit) and _tofit.min()
        delta = 0
        if _tofitmin <= 0:
            delta = -_tofitmin + 1e-9
        tofit = np.log( _tofit + delta)

        fitpara = np.polyfit(np.linspace(max(left_ips-fitwindow,t[0]),left_ips,len(tofit)),np.array(tofit,dtype=float),deg=degree)
        
        thresholdpara = fitpara + np.array( [0]*degree +[np.log(1-offset)])
        thresholdline = np.poly1d(thresholdpara) 
        _tosearch = findTimeVal(t,smoothed_c,left_ips,t[-1])
        _tosearchmin = len(_tosearch) and _tosearch.min()
        delta = 0
        if _tosearchmin <= 0:
            delta = -_tosearchmin + 1e-9
        tosearch = np.log(_tosearch + delta)
        tosearchT = np.linspace(left_ips,t[-1],len(tosearch))
        thresholdSearch = thresholdline(tosearchT) - tosearch
        thresholdCt = left_ips
        for sT,sthre in zip(tosearchT,thresholdSearch):        
            if sthre > 0:
                break
            thresholdCt = sT
        return  [*X[0:-3],*thresholdpara,left_ips]
          
    def transform(self,X,y=None):        
        return np.array([self.transformer(i) for i in X])

        
class HyperCt(BaseEstimator,TransformerMixin):
    """
    calculate the Ct from threshold method,
    the threshold line is from a hyperbolic fitting
    the fitting time range is from 5min to left peak base on the 2nd derivative peak.
    """
    def __init__(self,offset=0.05):
        """        
        offset is how much the fitted curve shifts down. this is in relative scale to the intial fitting point.
        """        
        self.offset = offset
                
    def fit(self,X,y=None):        
        return self    
    
    def hyper(self,p,x,y):
        return p[0]/(x+p[1]) +p[2] -y
    def hyperF(self,p):
        return lambda x:p[0]/(x+p[1]) +p[2]

    def transformer(self,X):        
        offset = self.offset
        t,deri,smoothed_c = X[-3:]
        left_ips,peak_prominence,peak_width = X[0:3]
        tofit = findTimeVal(t,smoothed_c,t[0],left_ips - t[0])
        
        fitres = least_squares(self.hyper,x0=[5,5,0.5],
                    args=(np.linspace(t[0],left_ips,len(tofit)),tofit)
                    )
        fitpara = fitres.x
        
        thresholdpara = fitpara - np.array([0,0,(tofit[-1]) * offset])
        
        thresholdline = self.hyperF(thresholdpara)
        tosearch = findTimeVal(t,smoothed_c,left_ips,t[-1])        
        tosearchT = np.linspace(left_ips,t[-1],len(tosearch))
        thresholdSearch = thresholdline(tosearchT) - tosearch
        thresholdCt = left_ips
        for sT,sthre in zip(tosearchT,thresholdSearch):        
            if sthre > 0:
                break
            thresholdCt = sT
        return  [*X[0:-3],*thresholdpara,left_ips]
          
    def transform(self,X,y=None):        
        return np.array([self.transformer(i) for i in X])

    
        
class LeftIpsCt(BaseEstimator,TransformerMixin):
    """
    use the leftips as ct for next step.
    """
    def fit(self,X,y=None):        
        return self    
    
    def hyper(self,p,x,y):
        return p[0]/(x+p[1]) +p[2] -y
    def hyperF(self,p):
        return lambda x:p[0]/(x+p[1]) +p[2]

    def transformer(self,X):            
        return  [*X[0:-3],X[0]]
          
    def transform(self,X,y=None):        
        return np.array([self.transformer(i) for i in X])

    
            
class CtPredictor(BaseEstimator,TransformerMixin):
    "a predictor to predict result based on ct and prominence threshold from FindPeak"
    def __init__(self,ct=18.9,prominence=0.2,sd=0.1):
        self.ct=ct
        self.prominence = prominence
        self.sd=sd
                
    def fit(self,X,y=None):        
        return self    
        
    def transformer(self,x):
        """
        return 0,1 flag, thresholdCt, prominence, signal drop at 5min
        """
        return int(x[-1]<=self.ct and x[1]>=self.prominence and x[8]>=self.sd),x[-1],x[1],x[5]
        
    def transform(self,X,y=None):        
        return np.apply_along_axis(self.transformer,1,X)
        
        
                
class SdPrPredictor(BaseEstimator,TransformerMixin):
    "a predictor to predict result based on ct and prominence threshold from FindPeak"
    def __init__(self,prominence=0.2,sd=0.106382 ):        
        self.prominence = prominence
        self.sd=sd
                
    def fit(self,X,y=None):        
        return self    
        
    def transformer(self,x):
        """
        return 0,1 flag, thresholdCt, prominence, signal drop at 5min
        """
        pr = x[1]
        sd = x[5]
        result = pr > self.prominence and sd > self.sd
        return int(result),x[-1],x[1],x[5]
        
    def transform(self,X,y=None):        
        return np.apply_along_axis(self.transformer,1,X)
        
class Normalize(BaseEstimator,TransformerMixin):
    """
    Transformer to normalize an array with given parameters
    params: 
    mode: str, can be 'max', 'mean', 
    dataTimeRange: float, describe the total length of data in minutes.
    normalzieToTrange: (), tuple, describe from and to time in minutes it will normalize to.
    
    """
    def __init__(self,mode='max',normalizeRange=(5,20)):
        self.mode=mode
        self.normalizeRange = normalizeRange
        self.q_ = {'max':np.max,'mean':np.mean}.get(self.mode,None)    
        self.from_ = self.normalizeRange [0] 
        self.to_ = self.normalizeRange [1] 
            
    def fit(self,X,y=None):        
        return self
        
    def transformer(self,X):
        
        time,pc = X
        f = np.abs(np.array(time) - self.from_).argmin()
        t = np.abs(np.array(time) - self.to_).argmin()                
        normalizer = max(self.q_(pc[f:t]), 1e-3)
        return time,np.array(pc)/normalizer
        
    def transform(self,X,y=None):        
        return np.array([self.transformer(i) for i in X],dtype='object')

        
class Truncate(BaseEstimator,TransformerMixin):
    """
    Transformer to Truncate and interpolate data, 
    input X is a time and current 2d array. 
    [0,0.3,0.6...] in minutes,
    [10,11,12...] current in uA.
    return a 1d data array, with n data points, start from cutoffStart time, 
    end at cutoffEnd time. Time are all in minutes.
    """
    def __init__(self,cutoffStart,cutoffEnd,n):
        self.cutoffStart = cutoffStart 
        self.cutoffEnd = cutoffEnd      
        self.n = n
    def fit(self,X,y=None):
        return self 
    def transformer(self,X,y=None):
        t,pc = X
        c = (self.cutoffEnd - self.cutoffStart) / (t[-1] - t[0]) * len(t)
        # i have to do this float conversion, otherwise I got dtype inexact problem in polyfit.
        newcurrent = extract_timepionts(np.array([float(i) for i in t]), 
                                        np.array([float(i) for i in pc]),self.cutoffStart,self.cutoffEnd,self.n)        
        return np.linspace(self.cutoffStart,self.cutoffEnd,int(c)),newcurrent
    def transform(self,X,y=None):
        return np.array([self.transformer(i) for i in X],dtype='object')

class RemoveTime(BaseEstimator,TransformerMixin):
    """
    Transformer to Truncate and interpolate data, 
    input X is a time and current 2d array. 
    [0,0.3,0.6...] in minutes,
    [10,11,12...] current in uA.
    return a 1d data array, with n data points, start from cutoffStart time, 
    end at cutoffEnd time. Time are all in minutes.
    """
    def __init__(self,):
        pass
    def fit(self,X,y=None):
        return self 
    def transformer(self,X):
        t,pc = X        
        return pc
    def transform(self,X,y=None):        
        return np.array([self.transformer(i) for i in X],dtype='object')


class SmoothScale(BaseEstimator,TransformerMixin):
    def __init__(self,**params):
        self.params=params
        self.scaler = StandardScaler()        
    def fit(self,X,y=None):
        smoothed = np.apply_along_axis(traceProcessPipe(**self.params),1,X,)
        self.scaler.fit(smoothed)
        return self
    def transform(self,X,y=None):
        smoothed = np.apply_along_axis(traceProcessPipe(**self.params),1,X,)
        return self.scaler.transform(smoothed)


def load_model(file):
    return joblib.load(file)

def save_model(clf,file):
    return joblib.dump(clf,file)


def train_model(transformer,model,X,y):
    clf = Pipeline([(transformer,transformers[transformer]()),
                    (model,algorithm[model](max_iter=10000))])
    clf.fit(X,y)    
    return clf

def cross_validation(clf,X,y,fold=5,):
    skfold = StratifiedKFold(n_splits=fold,random_state=42, shuffle=True)
    precision = []
    recall = []
    for train_idx, test_idx in skfold.split(X,y):
        cloneclf = clone(clf)
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_test_fold = X[test_idx]
        y_test_fold = y[test_idx]
        cloneclf.fit(X_train_fold,y_train_fold)    
        precision.append(precision_score(y_test_fold,cloneclf.predict(X_test_fold),))
        recall.append(recall_score(y_test_fold,cloneclf.predict(X_test_fold),))
    return precision,recall

transformers = {'S-T-N': SmoothTruncateNormalize,'Smooth-Scale':SmoothScale}

algorithm = {'LinearSVC':LinearSVC}




if __name__=='__main__':
   


    import json

    with open('Xdata.json') as f:
        X = json.load(f)
        


    with open('ydata.json') as f:
        y = json.load(f)
        
        
    X = convert_list_to_X(X)


    clf = train_model('Smooth','LinearSVC',X,y)

    save_model(clf,'model')


    clf.predict([X[0]])

    t = X[0][0]
    pc = X[0][1]



    clf.predict(convert_list_to_X([[t,pc]]))[0]==1

    X[0]