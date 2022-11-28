#%%
import numpy as np
import json
import os

def get_output(source, num, dig=5):
    sec_center=np.linspace(-np.pi,np.pi,num+1)
    output=np.zeros(num)
    sec_dis=2*np.pi/num
    #angle=np.arctan2(source[1],source[0])
    angle=np.arctan2(source[0]["position"][1],source[0]["position"][0])
    before_indx=int((angle+np.pi)/sec_dis)
    if before_indx>=num:
        before_indx-=num
    after_indx=before_indx+1
    if after_indx>=num:
        after_indx-=num
    w1=abs(angle-sec_center[before_indx])
    w2=abs(angle-sec_center[after_indx])
    if w2>sec_dis:
        w2=abs(angle-(sec_center[after_indx]+2*np.pi))
    output[before_indx]+=w2/(w1+w2)
    output[after_indx]+=w1/(w1+w2)
    if type(dig)==int:  #!
        for j in range(num):
            output[j]=round(output[j],dig)
    if int(np.sum(output))!=1:
        print('output_sum: ', np.sum(output))
        print(output)
        print(source)
    return output

def get_output_mul(sources, num, dig=5):
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
            #print w2
        output[before_indx]+=w2/(w1+w2)*ws[i]
        output[after_indx]+=w1/(w1+w2)*ws[i]

    if type(dig)==int:  #!
        for j in range(num):
            output[j]=round(output[j],dig)
    #print angle,sec_center[before_indx],sec_center[after_indx]
    return output

class Dataset(object):
    """docstring for Datasedt"""
    def __init__(self, seg_angles, output_fun,path, dig=5):    #!20220729
        super(Dataset, self).__init__()
        files=os.listdir(path)
        self.names=[]
        self.source_list=[]
        xdata=[]
        ydata=[]
        for filename in files:
            if not filename.endswith('.json'):continue
            with open(os.path.join(path,filename),'r') as f:
                data=json.load(f)
                self.names.append(filename)
                if type(dig)==int:  #!
                    xdata.append([round(d,dig) for d in data['input']])
                else:
                    xdata.append(data['input'])
                ydata.append(output_fun(data['source'], seg_angles))    #!20220729
                # source=data['source']
                self.source_list.append(data['source'])

        xdata=np.array(xdata)
        ydata=np.array(ydata)

        # xx=xdata
        # yy=ydata

        self.xdata=xdata
        self.ydata=ydata
        self.data_size=xdata.shape[0]

class Trainset(object):
    """docstring for Trainset"""
    def __init__(self, xdata,ydata,info=None,source_num=[2],prob=[1.]):
        super(Trainset, self).__init__()
        self.info=info
        self.xdata=xdata
        self.ydata=ydata
        self.ws=xdata.mean(axis=1)
        self.data_size=xdata.shape[0]
        self.x_size=xdata.shape[1]
        self.y_size=ydata.shape[1]
        self.index_list=np.arange(self.data_size)
        self.source_num=source_num
        self.prob=prob

    def get_batch(self,bs):
        source_num=self.source_num
        prob=self.prob
        xs=[]
        ys=[]
        for i in range(bs):
            x,y=self.get_one_data(source_num,prob)
            x=x.reshape(1,-1)
            y=y.reshape(1,-1)
            xs.append(x)
            ys.append(y)
            # #!20221127
            # data_y = y
            # print(f'[get_batch_forloop] Check process with {data_y.shape[0]} data!')
            # # for k in range(data_y.shape[0]):
            # suml = np.sum(data_y)
            # if suml <1:
            #     print(f'[get_batch_forloop] Sum error occurs with {i}th data: ', data_y)
            # #!20221127
            pass
        xx=np.concatenate(xs)
        yy=np.concatenate(ys)
        
        # #!20221127
        # data_y = yy
        # print(f'[get_batch] Check process with {data_y.shape[0]} data!')
        # for k in range(data_y.shape[0]):
        #     suml = np.sum(data_y[k,:])
        #     if suml <1:
        #         print(f'[get_batch] Sum error occurs with {k}th data: ', data_y[k,:])
        # #!20221127
        
        mm=xx[:,:].mean(axis=1,keepdims=True)
        vv=xx[:,:].var(axis=1,keepdims=True)
        mm=np.tile(mm,(1,xx.shape[1]))
        vv=np.tile(vv,(1,xx.shape[1]))
        xx=(xx-mm)/np.sqrt(vv)

        return xx,yy

    def get_batch_fixsource(self,bs,source_num):
        xs=[]
        ys=[]
        while True:
            if len(xs)>=bs:break
            x,y=self.get_one_data([source_num],[1.])
            if np.where(y!=0)[0].shape[0] != source_num*2:
                continue
            x=x.reshape(1,-1)
            y=y.reshape(1,-1)
            xs.append(x)
            ys.append(y)
            pass
        xx=np.concatenate(xs)
        yy=np.concatenate(ys)
        mm=xx[:,:].mean(axis=1,keepdims=True)
        vv=xx[:,:].var(axis=1,keepdims=True)
        mm=np.tile(mm,(1,xx.shape[1]))
        vv=np.tile(vv,(1,xx.shape[1]))
        xx=(xx-mm)/np.sqrt(vv)
        return xx,yy


    def get_one_data(self,source_num,prob):
        num=np.random.choice(source_num,p=np.array(prob,dtype=np.float64)/np.sum(prob))
        data_indx=np.random.choice(self.index_list,size=num)
        x=np.zeros(self.x_size)
        y=np.zeros(self.y_size)
        # print('=============')
        # print('[get_one_data] data_indx: ', data_indx)  #!20221129
        # print('[get_one_data] (xsize): ', self.x_size)   #!20221127
        # print('[get_one_data] (ysize): ', self.y_size)   #!20221127
        # print('[get_one_data] num: ', num)  #!20221129
        ws=0.
        for indx in data_indx:
            # print('[get_one_data] indx: ', indx)  #!20221129
            x+=self.xdata[indx,:]
            # ws+=self.ws[indx]
            # y+=self.ws[indx]*self.ydata[indx,:]
            y+=self.ydata[indx,:]/len(data_indx)
            # print('[get_one_data] (dx): ',self.xdata[indx,:])   #!20221127
            # print('[get_one_data] (dws): ',self.ws[indx])   #!20221127
            # print('[get_one_data] (dy): ',self.ydata[indx,:])   #!20221127
        # if ws !=0:
        #     y=y/ws
        # print('=============')
        return x,y

    def split(self,split_fold,indx,test_size=None,seed=None):
        source_num=self.source_num
        prob=self.prob
        sub_size=self.data_size/split_fold
        if test_size is None:
            if source_num==[1]:
                test_size=sub_size
            else:
                test_size=sub_size*2
        trains=[]
        start=indx*sub_size
        end=(indx+1)*sub_size
        if end>self.data_size:
            end=self.data_size
        test_x=self.xdata[start:end,:]
        test_y=self.ydata[start:end,:]
        test=Testset(test_x,test_y,test_size,seed,source_num,prob)
        train_xs=[]
        train_ys=[]
        for i in range(split_fold):
            if i == indx: continue
            start=i*sub_size
            end=(i+1)*sub_size
            if end>self.data_size:
                end=self.data_size
            train_xs.append(self.xdata[start:end,:])
            train_ys.append(self.ydata[start:end,:])
        train_x=np.concatenate(train_xs)
        train_y=np.concatenate(train_ys)
        train=Trainset(train_x,train_y,source_num=source_num,prob=prob)
        return train,test

class Testset(object):
    """docstring for Testset"""
    def __init__(self, xdata,ydata,test_size=None,seed=None,source_num=[2],prob=[1.]):
        super(Testset, self).__init__()
        #self.arg = arg
        self.xdata=xdata
        self.ydata=ydata
        self.ws=xdata.mean(axis=1)
        self.data_size_raw=xdata.shape[0]
        if source_num==[1] or test_size==None:
            self.data_size=self.data_size_raw
        else:
            self.data_size=test_size
        self.x_size=xdata.shape[1]
        self.y_size=ydata.shape[1]
        self.index_list=np.arange(self.data_size_raw)
        xx,yy=self.gen_data(source_num,prob,seed=seed)
        self.xdata=xx
        self.ydata=yy

    def gen_data(self,source_num,prob,seed=None):
        np.random.seed(seed)
        xs=[]
        ys=[]
        xx=self.xdata
        yy=self.ydata
        #?else:
        mm=xx[:,:].mean(axis=1,keepdims=True)
        vv=xx[:,:].var(axis=1,keepdims=True)
        mm=np.tile(mm,(1,xx.shape[1]))
        vv=np.tile(vv,(1,xx.shape[1]))
        xx=(xx-mm)/np.sqrt(vv)
        return xx,yy

    def get_one_data(self,source_num,prob):
        num=np.random.choice(source_num,p=np.array(prob,dtype=np.float64)/np.sum(prob))
        data_indx=np.random.choice(self.index_list,size=num)
        x=np.zeros(self.x_size)
        y=np.zeros(self.y_size)
        ws=0.
        for indx in data_indx:
            x+=self.xdata[indx,:]
            ws+=self.ws[indx]
            y+=self.ws[indx]*self.ydata[indx,:]
        if ws != 0:
            y=y/ws
        return x,y

    def get_batch(self,bs,indx):
        start=indx*bs
        end=(indx+1)*bs
        if end>self.data_size:
            end=self.data_size
        return self.xdata[start:end,:],self.ydata[start:end,:]

class FilterData2(object):
    """docstring for FilterData"""
    def __init__(self, filterpath, dig=5):
        super(FilterData2, self).__init__()
        self.path=filterpath
        filter_data=[]
        files=os.listdir(filterpath)
        filter_types = ['far', 'near']
        for i in filter_types:
            filter_data.append([])
        for filename in files:
            if not filename.endswith('.json'):continue
            with open(os.path.join(filterpath,filename),'r') as f:
                data=json.load(f)
                for i,filter_type in enumerate(filter_types):
                    if filter_type in filename:
                        if type(dig)==int:  #!
                            filter_data[i].append([round(d,dig) for d in data['input']])
                        else:
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


class FilterData(object):
    """docstring for FilterData"""
    def __init__(self, filterpath, dig=5):
        super(FilterData, self).__init__()
        self.path=filterpath
        filter_data=[]
        files=os.listdir(filterpath)
        for filename in files:
            if not filename.endswith('.json'):continue
            with open(os.path.join(filterpath,filename),'r') as f:
                data=json.load(f)
                if type(dig)==int:  #!
                    filter_data.append([round(d,dig) for d in data['input']])
                else:
                    filter_data.append(data['input'])
        filter_data=np.array(filter_data)

        mm=filter_data[:,:].mean(axis=1,keepdims=True)
        vv=filter_data[:,:].var(axis=1,keepdims=True)
        mm=np.tile(mm,(1,filter_data.shape[1]))
        vv=np.tile(vv,(1,filter_data.shape[1]))
        filter_data=(filter_data[:,:]-mm[:,:])/np.sqrt(vv[:,:])
        self.data=filter_data
        self.size=filter_data.shape[0]

def load_data(test_size,train_size,test_size_gen,seg_angles,output_fun,path,source_num,prob,seed):
    if test_size_gen is None:
        if source_num==[1]:
            test_size_gen=test_size
        else:
            test_size_gen=test_size*2

    data_set=Dataset(seg_angles,output_fun=output_fun,path=path)
    data_size=data_set.data_size
    
    #!20221127
    data_y = data_set.ydata
    print(f'Check process with {data_y.shape[0]} data!')
    for k in range(data_y.shape[0]):
        suml = np.sum(data_y[k,:])
        if suml <1:
            print(f'Sum error occurs with {k}th data: ', data_y[k,:])
    #!20221127
    
    if train_size is None:
        train_size=data_set.data_size-test_size
    train_set=Trainset(data_set.xdata[0:train_size,:],data_set.ydata[0:train_size,:],source_num=source_num,prob=prob)
    test_set=Testset(data_set.xdata[train_size:data_size,:],
            data_set.ydata[train_size:data_size,:],
            test_size_gen,source_num=source_num,prob=prob,seed=seed)
    return train_set,test_set


if __name__ == '__main__':
    path = 'openmc/discrete_2x2_100_data_20220728_v1.1'  #!20220716
    filterpath ='openmc/disc_filter_2x2_100_data_20220729_v1.1'    #!20220716
    filterdata=FilterData(filterpath)  
    filterdata2=FilterData2(filterpath)      
    path2 = 'openmc/discrete_10x10_128_data_20220803_v2.1'
    train_set,test_set=load_data(test_size=50,train_size=None,test_size_gen=None,seg_angles=100,output_fun=get_output,path=path,source_num=[1],prob=[1.],seed=None)
    train_set2,test_set2=load_data(test_size=50,train_size=None,test_size_gen=None,seg_angles=128,output_fun=get_output_mul,path=path2,source_num=[1, 1],prob=[1., 1.],seed=None)


# %%
