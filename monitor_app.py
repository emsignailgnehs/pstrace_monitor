import time
import os
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib
import json
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from tkinter import ttk
from file_monitor import  PSS_Handler,PSS_Logger,StartMonitor,save_csv,plot_curve_fit
from utils import timeseries_to_axis
import multiprocessing as mp
from collections import deque
from itertools import zip_longest
from utils import timeseries_to_axis
import glob
import pickle

matplotlib.use('TKAgg')

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("PSTrace master")
        
        self.load_settings() 
        self.tabs = ttk.Notebook(self)
        self.monitor = MonitorTab(parent=self.tabs, master=self)
        self.viewer = ViewerTab(parent=self.tabs,master=self)
        self.tabs.add(self.monitor,text = 'Monitor')
        self.tabs.add(self.viewer, text='Viewer')
        self.tabs.pack(expand=1,fill='both')
        self.tabs.bind('<<NotebookTabChanged>>',self.onNotebookTabChange)
        self.create_menus()
    
    def on_closing(self):
        "handle window closing. clean up shit"
        
        self.monitor.stop_monitor()
        while self.monitor.ismonitoring:
            time.sleep(0.1)

        if self.viewer.needToSave:
            confirm = tk.messagebox.askquestion('Unsaved data',
                "You have unsaved data, do you want to save?",icon='warning')
            if confirm=='yes':
                return 

        self.destroy()

    def onNotebookTabChange(self,e):
        selected = self.tabs.select()
        if selected == ".!notebook.!monitortab":
            self.geometry('900x660')
        elif selected == ".!notebook.!viewertab":
            self.geometry('1330x910')


    def create_menus(self):
        menu = tk.Menu(self)
        self.config(menu=menu)

        # file menu
        filemenu = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label='File', menu=filemenu)
        filemenu.add_command(label='New Monitor Folder', command=self.monitor.new_folder)
        filemenu.add_command(label='Save PStrace Edits', command=self.viewer.saveDataSource)
        filemenu.add_command(label='Quit', command=self.on_closing)

        # plot menu
        plotmenu = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label='Plot', menu=plotmenu)
        plotmenu.add_command(label='Plot Monitor Folder Curve Fit',
                             command=self.monitor.plot_curve_fit)
        
        # View Menu
        viewmenu = tk.Menu(menu,tearoff=False)
        menu.add_cascade(label='View', menu=viewmenu)
        viewmenu.add_command(label='Date View', command=self.viewer.switchView('dateView'))
        viewmenu.add_command(label='Experiment View', command=self.viewer.switchView('expView'))                


        # Pref menu  
        prefmenu = tk.Menu(menu,tearoff=False)
        menu.add_cascade(label='Preference',menu=prefmenu)
        prefmenu.add_command(label='Monitor Settings',command=self.edit_settings)
        prefmenu.add_command(label='Save Plot Settings',command=self.viewer.save_plot_settings)
        
        
    
    def edit_settings(self):
        "edit monitor settings"
        def submit():
            self.settings['PRINT_MESSAGES'] = printmsg.get()
            self.settings['MAX_SCAN_GAP'] = maxgap.get()
            self.settings['LOG_LEVEL'] = loglevel.get()
            self.save_settings()
            top.destroy()
            
        top = tk.Toplevel()
        top.title('Monitor Settings')
        
        printmsg = tk.BooleanVar()
        printmsg.set(self.settings['PRINT_MESSAGES'])
        tk.Label(top,text='Print Messages:').grid(row=0,column=0,padx=10,pady=10,sticky='e')
        tk.Radiobutton(top,text='True',variable=printmsg,value=True).grid(row=0,column=1,)
        tk.Radiobutton(top,text='False',variable=printmsg,value=False).grid(row=0,column=2,padx=10)


        maxgap = tk.IntVar()
        maxgap.set(self.settings['MAX_SCAN_GAP'])
        tk.Label(top,text='Max Scan Gap:').grid(row=1,column=0,padx=10,sticky=tk.E)
        tk.Entry(top,width=10,textvariable=maxgap).grid(row=1,column=1)

        loglevel = tk.StringVar()
        loglevel.set(self.settings['LOG_LEVEL'])
        tk.Label(top,text='Log Level:').grid(row=2,column=0,padx=10,pady=10,sticky=tk.E)
        tk.OptionMenu(top,loglevel,*['DEBUG','INFO','WARNING','ERROR','CRITICAL']).grid(row=2,column=1)

        subbtn = tk.Button(top, text='Save', command=submit)
        subbtn.grid(column=0, row=3,padx=10,pady=10)
        calbtn = tk.Button(top, text='Cancel', command=top.destroy)
        calbtn.grid(column=1,row=3,padx=10,pady=10)

    def load_settings(self):
        pp = (Path(__file__).parent / '.appconfig').absolute()
        if os.path.exists(pp):
            settings = json.load(open(pp, 'rt'))
        else:
            settings = dict(
                # default settings
                MAX_SCAN_GAP=30,  # mas interval to be considerred as two traces in seconds
                PRINT_MESSAGES=True,  # whether print message
                LOG_LEVEL='DEBUG',
                TARGET_FOLDER=str((Path(__file__).parent).absolute()),
                TreeViewFormat='dateView',
            )
        self.settings = settings

    def save_settings(self):
        pp = (Path(__file__).parent / '.appconfig').absolute()
        with open(pp, 'wt') as f:
            json.dump(self.settings, f, indent=2)


class MonitorTab(tk.Frame):
    def __init__(self, parent=None,master=None):
        super().__init__(parent)
        self.master = master
        self.settings = master.settings
        self.save_settings = master.save_settings
        self.create_widgets()
        self.create_figure()
        self.MONITORING = None
        self.plotData = []
        self.bind('<1>',lambda e: self.focus_set())
         
    @property
    def ismonitoring(self):
        return self.MONITORING and self.MONITORING.get('process', False) and self.MONITORING['process'].is_alive()

    def new_folder(self):
        self.settings['TARGET_FOLDER'] = tk.filedialog.askdirectory(
            initialdir=str(Path(self.settings['TARGET_FOLDER']).parent))
        self.folderinput.delete(0, tk.END)
        self.folderinput.insert(tk.END, self.settings['TARGET_FOLDER'])
        self.save_settings()

    def plot_curve_fit(self):
        interval = tk.simpledialog.askinteger("Plot Interval","Enter integer interval between 1 to 100.\n 15 is roughly 1min.",
                    parent=self.master,minvalue=1,maxvalue=100,initialvalue=15)
        # done = plot_curve_fit()
        target_folder = self.settings['TARGET_FOLDER']
        tf = Path(target_folder)
        pstraces_loc = tf / f'{tf.stem}_pstraces.pickle'
        if interval:
            if os.path.exists(pstraces_loc):
                if self.ismonitoring:
                    p1,p2 = mp.Pipe()
                    self.MONITORING['pipe'].send({'action':'senddata','pipe':p2})
                else:
                    p1 = None
                p = mp.Process(target=plot_curve_fit, args=(target_folder, interval,p1))
                p.start()
                self.displaymsg(f"Curve fit saved to <{target_folder}>")
            else:
                self.displaymsg(f"PStraces file <{pstraces_loc}> doesn't exist.")
        
    def create_figure(self):
        "make canvas for figures"
        figures = []
        self.axes = []
        self.canvas=[]
        self.trace_edit_tools = []
        for i in range(8):
            f = Figure(figsize=(2, 1.6), dpi=100)
            ax = f.subplots()
            figures.append(f)
            self.axes.append(ax)
            ax.set_xticks([])
            ax.set_yticks([])
            f.set_tight_layout(True)

        T = tk
        for i in range(8):
            row = i // 4
            col = i % 4
            canvas = FigureCanvasTkAgg(figures[i], self)
            self.canvas.append(canvas)
            # canvas.draw()
            tkwidget = canvas.get_tk_widget()
            tkwidget.grid(column=col*20, row=row*4+1, columnspan=20,  )
            tkwidget.bind('<1>', lambda e: self.focus_set())
            name = T.Label(self,text='Name')
            nameE = T.Entry(self, textvariable="", width=15)
            exp = T.Label(self, text='Exp')
            expE = T.Entry(self, textvariable="", width=15)
            save = tk.Button(self,text='Save Changes',command=self.trace_edit_cb(i))
            delete = tk.Button(self, text='X', fg='red',command=self.trace_delete_cb(i),)
            name.grid(column=col*20,row=row*4 + 2,columnspan= 2, sticky=tk.E)
            nameE.grid(column=col*20 + 2, row=row*4 + 2, columnspan=16, )
            exp.grid(column=col*20, row=row*4 + 3, columnspan=2, sticky=tk.E)
            expE.grid(column=col*20 + 2, row=row*4 + 3, columnspan=16, )
            save.grid(column=col*20+5,row=row*4 + 4, columnspan=15)
            delete.grid(column=col*20,row=row*4+4,columnspan=5)
            self.trace_edit_tools.append((nameE,expE,))

    def trace_delete_cb(self,id):
        "generate delete button callback"
        def func():
            if self.ismonitoring:
                if len(self.plotData) > id:
                    chanel = self.plotData[id]['chanel']
                    name = self.plotData[id]['name']
                    confirm = tk.messagebox.askquestion('Delete data!',
                        f'Delete {chanel} - {name} data?', icon='warning')
                    if confirm == 'yes':
                        idx = self.plotData[id]['idx']
                        pipe = self.MONITORING['pipe']
                        pipe.send({'action':'delete','chanel':chanel,'idx':idx})
            else:
                self.displaymsg('Not Monitoring!','yellow')
            
        return func

    def trace_edit_cb(self,id):
        "generate save chagnes button callback"
        def func():
            if self.ismonitoring:
                if len(self.plotData) > id:
                    chanel = self.plotData[id]['chanel']
                    idx = self.plotData[id]['idx']

                    pipe = self.MONITORING['pipe']
                    name = self.trace_edit_tools[id][0].get()
                    exp = self.trace_edit_tools[id][1].get()
                    pipe.send({'action': 'edit', 'chanel': chanel, 'idx': idx, 'name':name,'exp':exp})
            else:
                self.displaymsg('Not Monitoring!', 'yellow')

        return func

    def start_plotting(self):
        if self.ismonitoring:
            pipe = self.MONITORING['pipe']
            datatoplot = []
            while pipe.poll():
                datatoplot = pipe.recv()

            datatoplot = list(filter(lambda x: not x['deleted'],datatoplot)) 
            
            if datatoplot:
                for (od, nd), ax, canvas, tool in zip(zip_longest(self.plotData, datatoplot), self.axes, self.canvas, self.trace_edit_tools):
                    if (od and nd and od['chanel'] == nd['chanel'] and od['idx'] == nd['idx']
                        and od['name'] == nd['name'] and od['exp']==nd['exp'] 
                        and len(od['time']) == len(nd['time']) and od['color']==nd['color']):
                        # don't need to plot
                        continue 
                    elif (not nd):
                        ax.clear()
                    else:
                        t = nd['time']
                        c = nd['pc']
                        color = nd['color']
                        ax.clear()
                        ax.plot(t,c,marker='o',linestyle='',markersize=2,markerfacecolor='w',color=color)
                        ax.set_title(f"{nd['chanel']}",color=color,fontsize=8)
                        ax.tick_params(axis='x', labelsize=6)
                        ax.tick_params(axis='y', labelsize=6)
                        canvas.draw()
                        nameE, expE = tool
                        if (not od) or od['name'] != nd['name']:
                            nameE.delete(0, tk.END)
                            nameE.insert(tk.END, nd['name'])
                        if (not od) or od['exp'] != nd['exp']:
                            expE.delete(0, tk.END)
                            expE.insert(tk.END, nd['exp'])
                self.plotData = datatoplot
            self.displaymsg('Monitoring...', 'yellow')
            self.plotjob = self.after(1001,self.start_plotting)
        else:
            self.displaymsg('Monitor stopped.','cyan')

    def callback(self,id):
        print(id)
        def func(event):
            # event.widget.grid_forget()
            print(id, event)
            x,y = (event.x,event.y) 
            ax = self.axes[id]
            ax.plot([x],[y],marker='o')
            ax.set_title('plot')
            self.canvas[id].draw()
        return func

    def create_widgets(self):
        
        # self.pack(fill=tk.BOTH, expand=True)

        # first row
        self.folderbutton = tk.Button(
            self, text='Folder', command=self.new_folder)
        self.folderbutton.grid(row=0, column=0, padx=10, pady=10, sticky=tk.E)
        self.folderinput = tk.Entry(self, width=50)
        self.folderinput.insert(tk.END, self.settings['TARGET_FOLDER'])
        self.folderinput.grid(row=0, column=10,columnspan=40, sticky=tk.W)
        self.start_monitor_button = tk.Button(
            self, text="Start Monitor", command=self.start_monitor)
        self.start_monitor_button.grid(row=0, column=50, columnspan=10)
        self.stop_monitor_button = tk.Button(self, text="Stop Monitor", fg="red", state='disabled',
                                             command=self.stop_monitor)
        self.stop_monitor_button.grid(row=0, column=60, columnspan=10)
        self.save_csv_button = tk.Button(self, text="Save CSV", fg="green",
                                         command=self.save_csv)
        self.save_csv_button.grid(row=0, column=70, columnspan=10)

        self.msg = tk.StringVar()
        self.msg.set('PSS MONITOR READY')
        self.msglabel = tk.Label(self, textvariable=self.msg, bg='cyan')
        self.msglabel.grid(row=10, column=0, columnspan=80,pady=15)

    def displaymsg(self,msg,color='cyan'):
        self.msg.set(msg)
        if color:
            self.msglabel.config(bg=color)

    def save_csv(self):
        if self.ismonitoring:
            self.MONITORING['pipe'].send({'action':'savecsv'})
            saved = True
        else:
            saved = save_csv(self.settings['TARGET_FOLDER'])
        if saved:
            self.displaymsg(f"CSV Saved To {self.settings['TARGET_FOLDER']}!")
        else:
            self.displaymsg(f"No pstraces in {self.settings['TARGET_FOLDER']}",'red')

    def stop_monitor(self):
        # self.appPipe.send('stop')
        if self.ismonitoring:
            self.after_cancel(self.plotjob)
            while self.MONITORING['pipe'].poll():
                self.MONITORING['pipe'].recv()
            self.MONITORING['pipe'].send({'action':'stop'}) 
        self.plotData = []
        self.start_monitor_button['state'] = 'normal'
        self.folderinput['state'] = 'normal'
        # self.save_csv_button['state'] = 'normal'
        self.stop_monitor_button['state'] = 'disabled'
        self.folderbutton['state'] = 'normal'
        self.displaymsg('Monitor stopped.', 'cyan')
        

    def start_monitor(self, ):
        self.settings['TARGET_FOLDER'] = self.folderinput.get()

        if not os.path.exists(self.settings['TARGET_FOLDER']):
            self.msg.set(
                f"'{self.settings['TARGET_FOLDER']}' is not a valid folder.")
            self.msglabel.config(bg='red')
            return

        self.save_settings()
        self.start_monitor_button['state'] = 'disabled'
        self.folderinput['state'] = 'disabled'
        self.stop_monitor_button['state'] = 'normal'
        # self.save_csv_button['state'] = 'disabled'
        self.folderbutton['state'] = 'disabled'
        
        p,c = mp.Pipe()
        monitorprocess = mp.Process(target=StartMonitor, args=(self.settings,c))

        while self.ismonitoring:
            # wait until previous monitor is stopped. 
            time.sleep(0.1)

        self.MONITORING = {'process':monitorprocess,'pipe':p}
        monitorprocess.start()
        self.start_plotting()
        self.displaymsg('Monitoring...', 'yellow')
        
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

    def generate_treeview_menu(self,view='dateView'):
        "generate orderred data from self.pickles"
        Dataview = getattr(self,view)
        keys = list(Dataview.keys())
        keys.remove('deleted')
        if view == 'dateView':
            keys.sort(reverse=True)
            keys = [(k.strftime('%Y / %m / %d'), [ ( f"{k.strftime('%Y / %m / %d')}$%&$%&{idx}" , item['name']) for idx,item in enumerate(Dataview[k]) ]) for k in keys]
        elif view == 'expView':
            keys.sort()
            keys = [(k , [(f"{k}$%&$%&{idx}",item['name']) for idx,item in enumerate(Dataview[k])] ) for k in keys]
        
        keys.append(('deleted', [ (f"deleted$%&$%&{idx}" ,item['name']) for idx,item in enumerate(Dataview['deleted'])] ))
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


class ViewerTab(tk.Frame):
    defaultParams = {
            'color':'blue','linestyle': '-','marker':None,'label':"Curve",'alpha':0.75,
            'markersize': 1.0, 'linewidth': 0.5, 'ymin': 0.0, 'ymax': 100.0, 'markerfacecolor':'white',
            'markeredgecolor': 'black','title':'New Plot'
        }

    def __init__(self,parent=None,master=None):
        super().__init__(parent)
        self.master=master 
        self.settings = master.settings 
        self.save_settings = master.save_settings
        self.plot_state= PlotState(maxlen=200)
        
        self.datasource = ViewerDataSource()
        self.create_widgets()
        self.create_figures()
        self.bind('<1>', lambda e: self.focus_set()) 

        # development code:
        # rememb file history 
        history = self.settings.get('PStrace History',[])
        self.settings['PStrace History'] = [ i for i in history if os.path.exists(i)]
        self.datasource.load_picklefiles(self.settings['PStrace History'])
        # self.datasource.load_picklefiles(['/Users/hui/Downloads/2020-06-05/2020-06-05_pstraces.pickle'])
        self.updateTreeviewMenu()
    @property
    def needToSave(self):
        return self.datasource.needToSave

    def saveDataSource(self):
        self.save_settings()
        self.datasource.save()

    def create_widgets(self):
        ""
        scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL)
        xscrollbar = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        tree = ttk.Treeview(self, selectmode='extended', height=40, show=['tree'], yscrollcommand=scrollbar.set, xscrollcommand=xscrollbar.set,)
        tree.column("#0",minwidth=500,stretch=True)
        scrollbar.config(command=tree.yview)
        xscrollbar.config(command=tree.xview)

        tree.grid(column=0, row=1, padx=5, pady=5, rowspan=100, sticky='ns')
        scrollbar.grid(column=1, row=1, rowspan=100, sticky='nsw')
        xscrollbar.grid(column=0,row=101,sticky='we')
        self.tree = tree
        self.tree.bind('<<TreeviewSelect>>', self.treeviewselect_cb)

        tk.Button(self, text='Add PStrace', command=self.add_pstrace).grid(
            column=0, row=0, padx=(10,1), pady=(20,1), sticky='es')
        tk.Button(self, text='X',fg='red', command=self.drop_pstrace).grid(
            column=0, row=0, padx=(10,1), pady=(20,1), sticky='ws') 


        # information area 
        tk.Label(self, text='Name:').grid(column=11,row=35,sticky='w')
        tk.Label(self, text='Exp:').grid(column=11,row=36,sticky='w')
        tk.Label(self, text='Desc:').grid(column=11, row=37, sticky='nw')
        self.name = tk.Entry(self, textvariable="", width=22,)
        self.name.grid(column=11,row=35,columnspan=5,sticky='w',padx=(50,1))
        self.exp = tk.Entry(self, textvariable="", width=22,)
        self.exp.grid(column=11,row=36,columnspan=5,sticky='w',padx=(50,1))
        self.desc = tk.Text(self,  width=29, height=10, highlightthickness=2,undo=1)
        self.desc.configure(font=('Arial',12))
        self.desc.grid(column=11,row=37,columnspan=5,rowspan=10,sticky='w',padx=(50,1))
        
        self.name.bind('<FocusOut>',self.data_info_cb('name'))
        self.exp.bind('<FocusOut>',self.data_info_cb('exp'))
        self.desc.bind('<FocusOut>', self.data_info_cb('desc')) 

        # peak plot area 
        self.peak_start = tk.IntVar()
        tk.Label(self, text='Start:').grid(column=2,row=100,sticky='w',padx=15)
        tk.Entry(self, textvariable=self.peak_start, width=4 ).grid(column=2,row=100,padx=(40,1),)
        self.peak_gap = tk.IntVar()
        tk.Label(self, text='Gap:').grid(column=3, row=100, sticky='w',padx=5)
        tk.Entry(self, textvariable=self.peak_gap, width=4).grid(column=3, row=100, padx=(20, 1),)
        # self.peak_count = tk.IntVar()
        # self.peak_count.set(1)
        # tk.OptionMenu(self,self.peak_count,*[0,1,4]).grid(row=100,column=4,sticky='e',padx=(50,0))
        tk.Button(self,text=" < ",command=lambda : self.peak_start.set( max(self.peak_start.get() - 1,0) )).grid(column=4,row=100,sticky='w',padx=10)
        tk.Button(self, text=" > ",command=lambda : self.peak_start.set(self.peak_start.get() + 1 )).grid(column=4, row=100,padx=10)
        # self.peak_count.trace('w',self.peak_plot_params_cb('count',self.peak_count)) 
        self.peak_start.trace('w', self.variable_callback(self.peak_start,lambda *_: self.plotPeakFig(invodedFrom='peakparams')) )
        # self.peak_gap.trace('w', self.variable_callback(self.peak_gap,lambda *_: self.plotPeakFig(invodedFrom='peakparams')))


        # main plotting area tools 
        self.plot_params = {k:tk.StringVar() for k in [
            'color', 'linestyle', 'marker', 'label', 'markerfacecolor','markeredgecolor','title']}
        self.plot_params.update({k:tk.DoubleVar() for k in [
            'markersize','linewidth','ymin','ymax','alpha',
        ]})
        pp = self.plot_params

        self.init_plot_params()
        tk.Label(self,text='Plot Title:').grid(column=6,row=65,sticky='w',pady=7,padx=8)
        tk.Entry(self,textvariable=pp['title'],width=30).grid(column=6,row=65,columnspan=4,padx=(53,1))

        tk.Label(self,text='Legend').grid(column=6,row=66,columnspan=2)
        tk.Label(self,text='Y Min').grid(column=8,row=66)
        tk.Label(self,text='Y Max').grid(column=9,row=66)

        tk.Entry(self,textvariable=pp['label'],width=16).grid(column=6,row=67,columnspan=2)
        tk.Entry(self,textvariable=pp['ymin'],width=6).grid(column=8,row=67)
        tk.Entry(self,textvariable=pp['ymax'],width=6).grid(column=9,row=67)
        
        linecolors = ['blue','green','red','skyblue','orange','lime','royalblue','pink','cyan','white','black']
        tk.Label(self,text='Line Style').grid(column=6,row=68)
        tk.Label(self,text='Line Width').grid(column=7, row=68)
        tk.Label(self,text='Line Color').grid(column=8,row=68,padx=15)
        tk.Label(self,text='Line Alpha').grid(column=9,row=68,padx=15)
        tk.OptionMenu(self,pp['linestyle'],*[None,'-',':','--','-.',]).grid(column=6,row=69,sticky='we',padx=5)
        tk.Entry(self, textvariable=pp['linewidth'],width= 6 ).grid(column=7,row=69)
        tk.OptionMenu(self,pp['color'],*linecolors).grid(column=8,row=69,sticky='we',padx=5) 
        tk.Entry(self,textvariable=pp['alpha'],width=6).grid(column=9,row=69)

        markerstyle = [None] + list('.,ov^<>1+xs')
        tk.Label(self,text='Marker Style').grid(column=6,row=70)
        tk.Label(self,text='Marker Size').grid(column=7,row=70)
        tk.Label(self,text='Face Color').grid(column=8,row=70)
        tk.Label(self,text='Edge Color').grid(column=9,row=70)
        tk.OptionMenu(self, pp['marker'], *markerstyle).grid(column=6, row=71, sticky='we', padx=5)
        tk.Entry(self, textvariable=pp['markersize'], width=6).grid(column=7, row=71)
        tk.OptionMenu(self, pp['markerfacecolor'], *linecolors).grid(column=8, row=71, sticky='we', padx=5)
        tk.OptionMenu(self, pp['markeredgecolor'], *linecolors).grid(column=9, row=71, sticky='we', padx=5)
        
        self.undoBtn = tk.Button(self,text='Undo',command=self.undoMainPlot,state='disabled')
        self.undoBtn.grid(column=6,row=72,padx=10,pady=15)
        self.redoBtn = tk.Button(self,text='Redo',command=self.redoMainPlot,state='disabled')
        self.redoBtn.grid(column=7,row=72,padx=10,pady=15)
        tk.Button(self,text='Clear Plot',command=self.clearMainPlot).grid(column=8,row=72,padx=10,pady=15)
        tk.Button(self,text='Add To Plot',command=self.addMainPlot).grid(column=9,row=72,padx=10,sticky='we',pady=15)

    def switchView(self,view):
        def cb():
            self.settings['TreeViewFormat'] = view 
            self.updateTreeviewMenu()
        return cb

    @property 
    def TreeViewFormat(self):
        return self.settings.get('TreeViewFormat','dateView')

    def save_plot_settings(self):
        params = self.get_plot_params()
        self.settings.update(params)
        self.save_settings()

    def updateMainFig(self,datapacket):
        "draw additional data to figure without drawing yet"
        if datapacket == None:
            self.Max.clear() 
            return 
        data, params = datapacket 
        params = params.copy()
        ymin = params.pop('ymin')
        ymax = params.pop('ymax')
        title = params.pop('title')
        self.Max.set_title(title,)
        self.Max.set_ylim([ymin,ymax])
        self.Max.set_xlabel('Time / mins')
        self.Max.set_ylabel('Signal / nA')
        if data:
            t = timeseries_to_axis(data[0]['data']['time'])
            c = [i['pc'] for i in data[0]['data']['fit']]  
            self.Max.plot(t,c,**params) 
            params.pop('label')
            for d in data[1:]:
                t = timeseries_to_axis(d['data']['time'])
                c = [i['pc'] for i in d['data']['fit']]  
                self.Max.plot(t,c,**params)  
            self.Max.legend()

    def addMainPlot(self):
        data = self.getAllTreeSelectionData()
        params = self.get_plot_params() 
        if self.plot_state.isBack: 
        # if is back, use new data and new style or use old data for next plotting.
            data = data or self.plot_state.getNextData()[0] 
            self.updateMainFig((data,params)) 
            self.plot_state.advance()
            self.plot_state.updateCurrent((data,params))       
        else:
            # if is current, renew params if no data selected: 
            if not data:
                for packets in self.plot_state.fromLastClear():
                    self.updateMainFig(packets)
                data,_p = self.plot_state.getCurrentData()
                self.plot_state.updateCurrent((data,params))
            else:
                self.plot_state.append((data,params) )
            self.updateMainFig((data,params) )
        
        self.Mcanvas.draw()
        self.undoBtn['state'] = self.plot_state.undoState
        self.redoBtn['state'] = self.plot_state.redoState

    def clearMainPlot(self):
        self.plot_state.append(None)
        self.updateMainFig(None)
        self.Mcanvas.draw()
        self.undoBtn['state'] = self.plot_state.undoState
        self.redoBtn['state'] = self.plot_state.redoState

    def undoMainPlot(self):
        ""
        for packets in self.plot_state.fromLastClear():
            self.updateMainFig(packets)
        self.Mcanvas.draw()
        self.plot_state.backward()
        
        self.undoBtn['state'] = self.plot_state.undoState
        self.redoBtn['state'] = self.plot_state.redoState

    def redoMainPlot(self):
        ""
        self.updateMainFig(self.plot_state.getNextData())
        self.Mcanvas.draw()
        self.plot_state.advance()
        self.redoBtn['state'] = self.plot_state.redoState
        self.undoBtn['state'] = self.plot_state.undoState

    def variable_callback(self,var,callback):
        def wrap(*args,**kwargs):
            try:
                var.get()
                callback(*args,**kwargs)
            except:
                return 
        return wrap

    def init_plot_params(self):
        ""
        for k,var in self.plot_params.items():
            var.set(self.settings.get(k,self.defaultParams[k]))

    def data_info_cb(self, entry):
        def callback(e):
            if entry == 'desc':
                txt = e.widget.get(1.0,'end').strip()
            else:
                txt = e.widget.get().strip()
            if not txt:return 
            data,items = self.getAllTreeSelectionData(returnSelection=True)
            if not data: return 
            if data[0][entry] != txt:
                confirm = tk.messagebox.askquestion(f'Edit {entry}',
                f"Do you want to change <{entry}> on <{len(data)}> datasets??",icon='warning')
                if confirm != 'yes':
                    return 
            else:
                return 

            if entry=='name':
                if len(data) == 1:
                    self.datasource.modify(data[0],entry,txt)
                    self.tree.item(items[0],text=txt)
                else:
                    for i,(d,item) in enumerate(zip(data,items)):
                        nn = txt+'-'+str(i+1)
                        self.datasource.modify(d,entry,nn)
                        self.tree.item(item,text=nn) 
            else:
                for d in data:
                    self.datasource.modify(d,entry,txt)
                if entry=='exp': # need to rebuild menu 
                    self.datasource.rebuildExpView()
                    if self.TreeViewFormat == 'expView':
                        self.updateTreeviewMenu()

            
        return callback

    def get_plot_params(self):
        para = {}
        for k,i in self.plot_params.items():
            try:
                para[k]=i.get()
            except:
                para[k] = self.defaultParams[k]
        return para

    def plotBrowseFig(self):
        "plot Bfig"
        data = self.getAllTreeSelectionData()
        if not data: return 
        self.Bax.clear()
        params = self.get_plot_params()
        params.pop('label')
        params.pop('title')
        ymin = params.pop('ymin')
        ymax = params.pop('ymax')
        if len(data) == 1:
            name = data[0]['name'] 
        else:
            name = f'{len(data)} Curves'
        self.Bax.set_title(name)
        self.Bax.set_ylim([ymin,ymax])
        for d in data:
            t = timeseries_to_axis(d['data']['time'])
            c = [i['pc'] for i in d['data']['fit']]    
            self.Bax.plot(t,c,**params)
        self.Bcanvas.draw()

    def getAllTreeSelectionData(self,returnSelection=False):
        "get all data of current tree selection"
        currentselection = self.tree.selection()
        data = []
        selection = []
        for sele in currentselection:
            d = self.datasource.getData(sele, self.TreeViewFormat )
            if d : 
                data.append(d)
                selection.append(sele)
        if returnSelection: return data,selection
        return data

    def getFirstTreeSelectionData(self):
        "get the first data of current selection."
        currentselection = self.tree.selection()
        for sele in currentselection:
            data = self.datasource.getData(sele, self.TreeViewFormat)
            if data: return data
        return None

    def plotPeakFig(self,invodedFrom=None):
        "plot the first peak in selection"
        data = self.getFirstTreeSelectionData()
        if not data: return 
        # plot:
        name = data['name']
        self.Pfig.suptitle(name,fontsize=10)
        start = self.peak_start.get()
        interval = self.peak_gap.get()
        timeseries = timeseries_to_axis(data['data']['time'])
        times = timeseries[start::interval][0:4]
        raw = data['data']['rawdata'][start::interval][0:4]
        fit = data['data']['fit'][start::interval][0:4]
        for t,r,f,ax in zip(times,raw,fit,self.Pax):
            ax.clear()
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
                        fontsize=8, color=color) 
            ax.tick_params(axis='x',labelsize=7)
            ax.tick_params(axis='y', labelsize=7)
        self.Pcanvas.draw()

        if invodedFrom=='peakparams':
            # if this update is from peak params update, also change drawing in browser window. 
            self.Bax.clear()
            params = self.get_plot_params()
            params.pop('label')
            params.pop('title')
            ymin = params.pop('ymin')
            ymax = params.pop('ymax')
            self.Bax.set_title(name)
            self.Bax.set_ylim([ymin,ymax])
            # draw original plot
            self.Bax.plot(timeseries,[i['pc'] for i in data['data']['fit']],**params) 
            # markout current dots
            self.Bax.plot(times,[i['pc'] for i in fit],linestyle="", marker='x',markersize=8,color='red')
            self.Bcanvas.draw()

    def updatePeakVariables(self):
        "when select a new plot, only update peak gap to 1/4 of length, this won't trigger drawing peak."
        data = self.getFirstTreeSelectionData()
        if not data: return
        datalength = len(data['data']['time'])
        # self.peak_start.set(0)
        self.peak_gap.set((datalength-1)//4)

    def updateInfo(self):
        data = self.getFirstTreeSelectionData() 
        if  not data: return 
        self.name.delete(0,'end')
        self.name.insert('end',data['name'])
        self.exp.delete(0,'end')
        self.exp.insert('end',data['exp'])
        self.desc.delete(1.0,'end')
        self.desc.insert('end',data['desc'])

    def treeviewselect_cb(self,e):
        "call back for chaning treeview selection."
        self.updatePeakVariables()
        self.plotBrowseFig()
        # self.plotPeakFig()
        self.updateInfo()
       
    def create_figures(self):
        # main plot window
        self.Mfig = Figure(figsize=(8,5.5),dpi=90)
        self.Max  = self.Mfig.subplots() 
        self.Mfig.set_tight_layout(True)
        self.Mcanvas = FigureCanvasTkAgg(self.Mfig, self)
        self.Mcanvas.get_tk_widget().grid(column= 2,row= 0,columnspan = 9 , pady=15, padx=15, rowspan = 65, sticky='n' )
        self.Mcanvas.callbacks.connect('button_press_event',self.save_fig_cb(self.Mfig))

        # peaks window
        self.Pfig = Figure(figsize=(4,3),dpi=90)
        self.Pax = [i for j in self.Pfig.subplots(2, 2) for i in j]
        self.Pfig.set_tight_layout(True)
        self.Pcanvas = FigureCanvasTkAgg(self.Pfig, self)

        for pax in self.Pax:
            pax.set_xticks([])
            pax.set_yticks([])

        self.Pcanvas.get_tk_widget().grid(column=2,row=65,columnspan = 4 ,rowspan=35,padx=15,sticky='nw')
        self.Pcanvas.callbacks.connect('button_press_event',self.save_fig_cb(self.Pfig))
       
        # browser figure window: 
        self.Bfig = Figure(figsize=(3,2.5),dpi=90)
        self.Bax = self.Bfig.subplots()
        self.Bfig.set_tight_layout(True)
        self.Bcanvas = FigureCanvasTkAgg(self.Bfig,self) 
        # self.Bcanvas.draw()
        self.Bcanvas.get_tk_widget().grid(column=11,row=0,columnspan=5,rowspan=35,padx=10,pady=10,sticky='n')

    def save_fig_cb(self,fig):
        def cb(e):
            if e.dblclick:
                files = [('All files','*'),('PNG image','*.png'),('SVG image','*.svg',),]
                file = tk.filedialog.asksaveasfilename(title='Save figure',filetypes=files,
                initialdir=self.datasource.picklefolder,)
                if file:
                    fig.savefig(file,dpi=150)
        return cb 

    def updateTreeviewMenu(self):
        ""
        for i in self.tree.get_children():
            self.tree.delete(i) 
        for parent, children in self.datasource.generate_treeview_menu(view=self.TreeViewFormat):
            self.tree.insert("",'end',parent, text=parent )
            for idx,childname in children:
                self.tree.insert(parent, 'end', idx, text=childname)
   
    def add_pstrace(self):
        "clear plot_state"
        selectdir = tk.filedialog.askdirectory(initialdir=str(
            Path(self.settings['TARGET_FOLDER']).parent))
        if selectdir:
            picklefiles = glob.glob(os.path.join(selectdir,'*pstraces.pickle'))
            if picklefiles:
                self.datasource.load_picklefiles(picklefiles)
                self.updateTreeviewMenu()
                self.settings['PStrace History'].extend(picklefiles)
         
        
    def drop_pstrace(self):    
        data = self.getAllTreeSelectionData()
        if not data: return 
        for d in data:
            self.datasource.modify(d,'deleted', (not d.get('deleted',False)) )
        self.datasource.rebuildDateView()
        self.datasource.rebuildExpView()
        self.updateTreeviewMenu()
        


if __name__ == "__main__":
    mp.set_start_method('spawn')
    app = Application()
    app.protocol('WM_DELETE_WINDOW',app.on_closing)
    app.mainloop()
