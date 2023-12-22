#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
import os
import openmc
from copy import copy

def gen_materials(panel_density):
    panel = openmc.Material(name='CdZnTe')
    panel.set_density('g/cm3', panel_density)
    panel.add_nuclide('Cd114', 33, percent_type='ao')
    panel.add_nuclide('Zn64', 33, percent_type='ao')
    panel.add_nuclide('Te130', 33, percent_type='ao')
    insulator = openmc.Material(name='Zn')
    insulator.set_density('g/cm3', 1)
    insulator.add_nuclide('Pb208', 11.35)
    outer = openmc.Material(name='Outer_CdZnTe')
    outer.set_density('g/cm3', panel_density)
    outer.add_nuclide('Cd114', 33, percent_type='ao')
    outer.add_nuclide('Zn64', 33, percent_type='ao')
    outer.add_nuclide('Te130', 33, percent_type='ao')
    materials = openmc.Materials(materials=[panel, insulator, outer])
    materials.export_to_xml()
    return panel, insulator, outer

def gen_settings(src_energy, src_strength, en_prob, num_particles, batch_size, sources):
    num_sources = len(sources)
    sources_list = []
    for i in range(num_sources):
        point = openmc.stats.Point((sources[i]['position'][0], sources[i]['position'][1], 0))
        source = openmc.Source(space=point, particle='photon', energy=src_energy, strength=src_strength)  #!20220204    #!20220118
        source.energy = openmc.stats.Discrete(x=(sources[i]['counts']), p=en_prob)
        sources_list.append(source)

    settings = openmc.Settings()
    settings.run_mode = 'fixed source'
    settings.photon_transport = True
    settings.source = sources_list
    settings.batches = batch_size
    settings.inactive = 10
    settings.particles = num_particles

    settings.export_to_xml()

def get_sources(sources_d_th):
    num_sources = len(sources_d_th)
    sources = []
    for i in range(num_sources):
        theta=sources_d_th[i][1]*np.pi/180
        dist = sources_d_th[i][0]
        source = {}
        src_xy = [float(dist*np.cos(theta)), float(dist*np.sin(theta))]
        source['position']=src_xy
        source['counts']=sources_d_th[i][2]
        sources.append(source)
    return sources

def run_openmc():
    # Run OpenMC!
    openmc.run()


def output_process(mean, digits, folder, file, sources, seg_angles, norm, savefig=False):
    mean = np.transpose(mean)
    if norm:
        mean_me = mean.mean()
        mean_st = mean.std()
        mean = (mean-mean_me)/mean_st
    num_sources = len(sources)
    data_json={}
    data_json['source']=sources
    data_json['output']=[round(s, digits) for s in get_output(sources, seg_angles).tolist()]
    data_json['num_sources']=num_sources
    data_json['seg_angles']=seg_angles
    mean_list=mean.T.reshape((1, -1)).tolist()
    data_json['input']=[round(m, digits) for m in mean_list[0]]
    with open(f"{folder}/{file}.json","w") as f:
        json.dump(data_json, f)
    if savefig:
        folder2 = folder + '_fig'
        mean_show  = np.flip(copy(mean), 0)
        plt.imshow(mean_show, interpolation='nearest', cmap='gist_gray')
        ds_ag_list = file.split('_')[1:]
        ds_ag_title = ''
        for i in range(num_sources):
            ds, ag = ds_ag_list[2*i], ds_ag_list[2*i+1]
            ds_ag_line = f'dist{i}: {ds},  angle{i}: {ag}'
            if i != num_sources-1:
                ds_ag_line += '\n'
            ds_ag_title += ds_ag_line
        plt.title(ds_ag_title)
        plt.xlabel('y')
        plt.ylabel('x')
        plt.colorbar()
        plt.savefig(f"{folder2}/{file}.png")
        plt.savefig(f"{folder2}/{file}.pdf")
        plt.close()
    return mean


def get_output(sources, num):
    sec_center=np.linspace(-np.pi,np.pi,num+1)
    output=np.zeros(num)
    sec_dis=2*np.pi/num
    ws=np.array([np.mean(source["counts"]) for source in sources])
    ws=ws/ws.sum()
    for i in range(len(sources)):
        source =sources[i]
        angle=np.arctan2(source["position"][1],source["position"][0])
        before_indx=int((angle+np.pi)/sec_dis)
        after_indx=before_indx+1
        if after_indx>=num:
            after_indx-=num
        w1=abs(angle-sec_center[before_indx])
        w2=abs(angle-sec_center[after_indx])
        if w2>sec_dis:
            w2=abs(angle-(sec_center[after_indx]+2*np.pi))
        output[before_indx]+=w2/(w1+w2)*ws[i]
        output[after_indx]+=w1/(w1+w2)*ws[i]
    return output

class Dataset(object):
    """docstring for Datasedt"""
    def __init__(self, seg_angles, output_fun,path):
        super(Dataset, self).__init__()
        files=os.listdir(path)
        self.names=[]
        self.source_list=[]
        xdata=[]
        ydata=[]
        zdata=[]
        for filename in files:
            if not filename.endswith('.json'):continue
            if filename.startswith('source'):continue
            with open(os.path.join(path,filename),'r') as f:
                data=json.load(f)
                self.names.append(filename)
                xdata.append(data['input'])
                ydata.append(output_fun(data['source'], seg_angles))
                zdata.append([float(da) for da in filename[:-5].split("_")[1:]])
                self.source_list.append(data['source'])
                

        xdata=np.array(xdata)
        ydata=np.array(ydata)
        zdata=np.array(zdata)

        self.xdata=xdata
        self.ydata=ydata
        self.zdata=zdata
        self.data_size=xdata.shape[0]

class Trainset(object):
    """docstring for Trainset"""
    def __init__(self, xdata,ydata,zdata,info=None,source_num=[2],prob=[1.]):
        super(Trainset, self).__init__()
        self.info=info
        self.xdata=xdata
        self.ydata=ydata
        self.zdata=zdata
        self.ws=xdata.mean(axis=1)
        self.data_size=xdata.shape[0]
        self.x_size=xdata.shape[1]
        self.y_size=ydata.shape[1]
        self.z_size=zdata.shape[1]
        self.index_list=np.arange(self.data_size)
        self.source_num=source_num
        self.prob=prob

    def get_batch(self,bs):
        source_num=self.source_num
        prob=self.prob
        xs=[]
        ys=[]
        zs=[]
        for i in range(bs):
            x,y,z=self.get_one_data(source_num,prob)
            x=x.reshape(1,-1)
            y=y.reshape(1,-1)
            z=z.reshape(1,-1)
            xs.append(x)
            ys.append(y)
            zs.append(z)
            pass
        xx=np.concatenate(xs)
        yy=np.concatenate(ys)
        zz=np.concatenate(zs)
        
        mm=xx[:,:].mean(axis=1,keepdims=True)
        vv=xx[:,:].var(axis=1,keepdims=True)
        mm=np.tile(mm,(1,xx.shape[1]))
        vv=np.tile(vv,(1,xx.shape[1]))
        xx=(xx-mm)/np.sqrt(vv)

        return xx,yy,zz

    def get_batch_fixsource(self,bs,source_num):
        xs=[]
        ys=[]
        zs=[]
        while True:
            if len(xs)>=bs:break
            x,y,z=self.get_one_data([source_num],[1.])
            if np.where(y!=0)[0].shape[0] != source_num*2:
                continue
            x=x.reshape(1,-1)
            y=y.reshape(1,-1)
            z=z.reshape(1,-1)
            xs.append(x)
            ys.append(y)
            zs.append(z)
            pass
        xx=np.concatenate(xs)
        yy=np.concatenate(ys)
        zz=np.concatenate(zs)
        mm=xx[:,:].mean(axis=1,keepdims=True)
        vv=xx[:,:].var(axis=1,keepdims=True)
        mm=np.tile(mm,(1,xx.shape[1]))
        vv=np.tile(vv,(1,xx.shape[1]))
        xx=(xx-mm)/np.sqrt(vv)
        return xx,yy,zz


    def get_one_data(self,source_num,prob):
        num=np.random.choice(source_num,p=np.array(prob,dtype=np.float64)/np.sum(prob))
        data_indx=np.random.choice(self.index_list,size=num)
        x=np.zeros(self.x_size)
        y=np.zeros(self.y_size)
        z=np.zeros(self.z_size)
        for indx in data_indx:
            x+=self.xdata[indx,:]
            y+=self.ydata[indx,:]/len(data_indx)
            z+=self.zdata[indx,:]
        return x,y,z
    def split(self,split_fold,step,test_size=None,seed=None):
        source_num=self.source_num
        prob=self.prob
        sub_size=self.data_size//split_fold
        if test_size is None:
            if source_num==[1]:
                test_size=sub_size
            else:
                test_size=sub_size*2
        trains=[]
        indx=step % split_fold
        start=indx*sub_size
        end=(indx+1)*sub_size
        if end>self.data_size:
            end=self.data_size
        test_x=self.xdata[start:end,:]
        test_y=self.ydata[start:end,:]
        test_z=self.zdata[start:end,:]
        test=Testset(test_x,test_y,test_z,test_size,seed,source_num,prob)
        train_xs=[]
        train_ys=[]
        train_zs=[]
        for i in range(split_fold):
            if i == indx: continue
            start=i*sub_size
            end=(i+1)*sub_size
            if end>self.data_size:
                end=self.data_size
            train_xs.append(self.xdata[start:end,:])
            train_ys.append(self.ydata[start:end,:])
            train_zs.append(self.zdata[start:end,:])
        train_x=np.concatenate(train_xs)
        train_y=np.concatenate(train_ys)
        train_z=np.concatenate(train_zs)
        train=Trainset(train_x,train_y,train_z,source_num=source_num,prob=prob)
        return train,test

class Testset(object):
    """docstring for Testset"""
    def __init__(self, xdata,ydata,zdata,test_size=None,seed=None,source_num=[2],prob=[1.]):
        super(Testset, self).__init__()
        #self.arg = arg
        self.xdata=xdata
        self.ydata=ydata
        self.zdata=zdata
        self.ws=xdata.mean(axis=1)
        self.data_size_raw=xdata.shape[0]
        if source_num==[1] or test_size==None:
            self.data_size=self.data_size_raw
        else:
            self.data_size=test_size
        self.x_size=xdata.shape[1]
        self.y_size=ydata.shape[1]
        self.z_size=zdata.shape[1]
        self.index_list=np.arange(self.data_size_raw)
        xx,yy,zz=self.gen_data(source_num,prob,seed=seed)
        self.xdata=xx
        self.ydata=yy
        self.zdata=zz

    def gen_data(self,source_num,prob,seed=None):
        np.random.seed(seed)
        xs=[]
        ys=[]
        zs=[]
        xx=self.xdata
        yy=self.ydata
        zz=self.zdata
        #?else:
        mm=xx[:,:].mean(axis=1,keepdims=True)
        vv=xx[:,:].var(axis=1,keepdims=True)
        mm=np.tile(mm,(1,xx.shape[1]))
        vv=np.tile(vv,(1,xx.shape[1]))
        xx=(xx-mm)/np.sqrt(vv)
        return xx,yy,zz

    def get_one_data(self,source_num,prob): #?
        num=np.random.choice(source_num,p=np.array(prob,dtype=np.float64)/np.sum(prob))
        data_indx=np.random.choice(self.index_list,size=num)
        x=np.zeros(self.x_size)
        y=np.zeros(self.y_size)
        z=np.zeros(self.z_size)

        ws=0.
        for indx in data_indx:
            x+=self.xdata[indx,:]
            ws+=self.ws[indx]
            y+=self.ws[indx]*self.ydata[indx,:]
            z+=self.zdata[indx,:]
        return x,y,z

    def get_batch(self,bs,indx):
        start=indx*bs
        end=(indx+1)*bs
        if end>self.data_size:
            end=self.data_size
        return self.xdata[start:end,:],self.ydata[start:end,:],self.zdata[start:end,:]

class FilterData2(object):
    """docstring for FilterData"""
    def __init__(self, filterpath):
        super(FilterData2, self).__init__()
        self.path=filterpath
        filter_data=[]
        files=os.listdir(filterpath)
        filter_types = ['far', 'near']
        for i in filter_types:
            filter_data.append([])
        for filename in files:
            if not filename.endswith('.json'):continue
            if filename.startswith('source'):continue
            with open(os.path.join(filterpath,filename),'r') as f:
                data=json.load(f)
                for i,filter_type in enumerate(filter_types):
                    if filter_type in filename:
                        filter_data[i].append(data['input'])
        filter_data=np.array(filter_data)
        filter_data = filter_data.reshape((-1,filter_data.shape[-1]))
        mm=filter_data[:,:].mean(axis=1,keepdims=True)
        vv=filter_data[:,:].var(axis=1,keepdims=True)
        mm=np.tile(mm,(1,filter_data.shape[1]))
        vv=np.tile(vv,(1,filter_data.shape[1]))
        filter_data=(filter_data[:,:]-mm[:,:])/np.sqrt(vv[:,:])

        self.data=filter_data
        self.size=filter_data.shape[0]

class FilterData1(object):
    """docstring for FilterData"""
    def __init__(self, filterpath, header=None):
        super(FilterData1, self).__init__()
        self.path=filterpath
        filter_data=[[]]
        files=os.listdir(filterpath)
        for filename in files:
            if header is not None:
                if not filename.startswith(header):continue
            if not filename.endswith('.json'):continue
            if filename.startswith('source'):continue
            with open(os.path.join(filterpath,filename),'r') as f:
                data=json.load(f)

                filter_data[0].append(data['input'])
        filter_data=np.array(filter_data)
        filter_data = filter_data.reshape((-1,filter_data.shape[-1]))
        mm=filter_data[:,:].mean(axis=1,keepdims=True)
        vv=filter_data[:,:].var(axis=1,keepdims=True)
        mm=np.tile(mm,(1,filter_data.shape[1]))
        vv=np.tile(vv,(1,filter_data.shape[1]))
        filter_data=(filter_data[:,:]-mm[:,:])/np.sqrt(vv[:,:])

        self.data=filter_data
        self.size=filter_data.shape[0]

def load_data(test_ratio,test_size_gen,seg_angles,output_fun,path,source_num,prob,seed):
    data_set=Dataset(seg_angles,output_fun=output_fun,path=path)
    data_size=data_set.data_size
    test_size = int(test_ratio*data_size)
    train_size=data_size-test_size
    if test_size_gen is None:
        test_size_gen=test_size

    data_y = data_set.ydata
    print(f'Total datasize: {data_y.shape[0]}')
    for k in range(data_y.shape[0]):
        suml = np.sum(data_y[k,:])
        if suml <0.5:
            print(f'Sum error occurs with {k}th data: ', data_y[k,:])
    if train_size is None:
        train_size=data_set.data_size-test_size
    idx_train, idx_test = train_test_split(range(data_set.data_size), test_size=test_size, random_state=seed)
    
    train_set=Trainset(data_set.xdata[idx_train,:],data_set.ydata[idx_train,:],data_set.zdata[idx_train,:],source_num=source_num,prob=prob)
    test_set=Testset(data_set.xdata[idx_test,:],data_set.ydata[idx_test,:],data_set.zdata[idx_test,:],
            test_size_gen,source_num=source_num,prob=prob,seed=seed)
    return train_set,test_set


def compute_accuracy(real, pred):
    real = real.detach().cpu()
    pred = pred.detach().cpu()
    diff = abs(real - pred)
    accuracy = 1.0 - (diff.sum() / diff.size(0))
    return accuracy

# %%
