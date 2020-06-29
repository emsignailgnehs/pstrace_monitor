import numpy as np
from matplotlib.figure import Figure
import pickle
from datetime import datetime
from pathlib import Path

def timeseries_to_axis(timeseries):
    "convert datetime series to time series in minutes"
    return [(d-timeseries[0]).seconds/60 for d in timeseries]

def plot_experiment(dataset, interval, savepath):
    """
    plot data into grid axes,
    dataset should be the format of mongodb datapack.
    {
        name:,
        desc:,
        exp:,
        dtype: 'covid=trace',
        data:{
            time: [datetime(),...]
            rawdata: [[v,a]...]
            fit:[ {'fx': , 'fy': , 'pc': , 'pv': , 'err': 0}...]
        }
    }
    interval: the interval for timepoints to be plotted.
    savepath: folder to save the file.
    """
    times = timeseries_to_axis(dataset['data']['time'][::interval])
    raw = dataset['data']['rawdata'][::interval]
    fit = dataset['data']['fit'][::interval]

    cols = int(np.ceil(np.sqrt(len(times))))
    rows = int(np.ceil(len(times) / cols))

    fig = Figure(figsize=(1.5*cols, 1.5*rows))
    axes = fig.subplots(rows, cols)
    axes = np.ravel([axes])

    for ax in axes:
        ax.axis('off')

    for t, r, f, ax in zip(times, raw, fit, axes):
        x1, x2 = f['fx']
        y1, y2 = f['fy']
        peakvoltage = f['pv']
        peakcurrent = f['pc']
        k = (y2-y1)/(x2-x1)
        b = -k*x2 + y2
        baselineatpeak = k * f['pv'] + b
        v, a = r
        color = 'r' if f['err'] else 'b'
        ax.plot(v, a,  f['fx'], f['fy'],
                [peakvoltage, peakvoltage], [baselineatpeak, baselineatpeak+peakcurrent])
        ax.set_title("{:.1f}m {:.2f}nA".format(t, peakcurrent),
                     fontsize=10, color=color)
        ax.axis('on')

    fig.set_tight_layout(True)

    fig.savefig(savepath)

def calc_peak_baseline(f):
    x1, x2 = f['fx']
    y1, y2 = f['fy']
    k = (y2-y1)/(x2-x1)
    b = -k*x2 + y2
    return k * f['pv'] + b


class ViewerDataSource():
    def __init__(self):
        """
        self.pickles: {'file': {'data': pickledata file, 'modified': True/False}}
        self.dateView: {datetime in days: [ orderred experiment data {name: exp: ...}], 'deleted': []}
        """
        self.pickles = {}
        self.dateView = {'deleted':[]}
        self.expView = {'deleted':[]}
        self.picklefolder = ""
    @property
    def needToSave(self):
        for f,d in self.pickles.items():
            if d['modified']:
                return True
        return False

    def save(self):
        for f,d in self.pickles.items():
            if d['modified']:
                with open(f,'wb') as o:
                    pickle.dump(d['data'],o)
                d['modified'] = False

    def load_picklefiles(self,files):
        for file in files:
            with open(file, 'rb') as f:
                data = pickle.load(f)
            # newdata.append((file,data))
            self.pickles[file] = {'data':data,'modified':False}
            self.picklefolder = Path(file).parent
        self.rebuildDateView()
        self.rebuildExpView()

    def modify(self,d,key,value):
        d[key]=value
        self.pickles[d['_file']]['modified'] = True

    def rebuildDateView(self):
        ""
        self.dateView = {'deleted':[]}
        for file,data in self.pickles.items():
            dataset = data['data']['pstraces']
            for _, cdata in dataset.items(): # cdata is the list of chanel data
                for edata in cdata: # edata is each dictionary of a timeseries tracing.
                    date = edata['data']['time'][0].replace(hour=0,minute=0,second=0)
                    deleted = edata.get('deleted',False)
                    edata['_file'] = file
                    # update new data to folder view
                    if deleted:
                        self.dateView['deleted'].append(edata)
                        continue
                    if date in self.dateView:
                        self.dateView[date].append(edata)
                    else:
                        self.dateView[date] = [edata]
        # sort new views by date
        for k,item in self.dateView.items():
            item.sort(key = lambda x: x['data']['time'][0])

    def rebuildExpView(self):
        ""
        self.expView = {'deleted':[]}
        for file,data in self.pickles.items():
            dataset = data['data']['pstraces']
            for _, cdata in dataset.items(): # cdata is the list of chanel data
                for edata in cdata: # edata is each dictionary of a timeseries tracing.
                    exp = edata['exp'] if edata['exp'] else 'Unassigned'
                    deleted = edata.get('deleted',False)
                    edata['_file'] = file
                    if deleted:
                        self.expView['deleted'].append(edata)
                        continue

                    if exp in self.expView:
                        self.expView[exp].append(edata)
                    else:
                        self.expView[exp] = [edata]
        # sort new views by date.
        for k,item in self.expView.items():
            item.sort(key = lambda x: x['data']['time'][0])

    def itemDisplayName(self,item):
        return item['name'] + item.get('_uploaded',False) * " ✓"

    def generate_treeview_menu(self,view='dateView'):
        "generate orderred data from self.pickles"
        Dataview = getattr(self,view)
        keys = list(Dataview.keys())
        keys.remove('deleted')
        if view == 'dateView':
            keys.sort(reverse=True)
            keys = [(k.strftime('%Y / %m / %d'), [ ( f"{k.strftime('%Y / %m / %d')}$%&$%&{idx}" ,
            self.itemDisplayName(item) ) for idx,item in enumerate(Dataview[k]) ]) for k in keys]
        elif view == 'expView':
            keys.sort()
            keys = [(k , [(f"{k}$%&$%&{idx}", self.itemDisplayName(item) )
            for idx,item in enumerate(Dataview[k])] ) for k in keys]

        keys.append(('deleted', [ (f"deleted$%&$%&{idx}" ,self.itemDisplayName(item))
         for idx,item in enumerate(Dataview['deleted'])] ))
        return keys

    def getData(self,identifier,view,):
        "get data from view with identifier"
        res = identifier.split('$%&$%&')
        if len(res) != 2:
            return None
        key,idx = res
        if view=='dateView':
            key = datetime.strptime(key ,'%Y / %m / %d') if key!='deleted' else key
        return getattr(self,view)[key][int(idx)]

class PlotState(list):
    def __init__(self,maxlen,):
        self.maxlen=maxlen
        self.current = 0
        super().__init__([None])
    @property
    def isBack(self):
        return len(self)-1 != self.current
    @property
    def undoState(self):
        if self.current <= 0:
            return 'disabled'
        else: return 'normal'
    @property
    def redoState(self):
        if self.current >= len(self)-1:
            return 'disabled'
        else: return 'normal'

    def updateCurrent(self,ele):
        self[self.current] = ele

    def getNextData(self):
        return self[self.current+1]

    def getCurrentData(self):
        return self[self.current]

    def append(self,ele):
        del self[self.current+1:]
        super().append(ele)
        if len(self)>self.maxlen:
            if None in self:
                idx = self.index(None)
            else:
                idx = len(self) // 2
            del self[:idx]
        self.current = len(self) - 1

    def advance(self,steps=1):
        self.current+=steps
        self.current = min(len(self)-1,self.current)

    def backward(self,steps=1):
        self.current-=steps
        self.current = max(0,self.current)

    def fromLastClear(self):
        steps = []
        for s in self[self.current-1::-1]:
            steps.append(s)
            if s==None:
                break
        return steps[::-1]
