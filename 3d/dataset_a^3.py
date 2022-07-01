#%%
import numpy as np
import json
import os

def get_output(source):
    sec_center=np.linspace(-np.pi,np.pi,41)
    sec_th=np.linspace(np.pi/36,np.pi*35/36,18)
    output_ph=np.zeros(40)
    output_th=np.zeros(18)
    sec_dis_ph=2*np.pi/40.
    sec_dis_th=np.pi/18.
    angle_ph=np.arctan2(source[1],source[0])
    angle_th=np.arctan2(np.sqrt(source[0]**2+source[1]**2), source[2])
    before_indx=int((angle_ph+np.pi)/sec_dis_ph)
    if before_indx>=40:
        before_indx-=40
    after_indx=before_indx+1
    if after_indx>=40:
        after_indx-=40
    w1=abs(angle_ph-sec_center[before_indx])
    w2=abs(angle_ph-sec_center[after_indx])
    if w2>sec_dis_ph:
        w2=abs(angle_ph-(sec_center[after_indx]+2*np.pi))
    output_ph[before_indx]+=w2/(w1+w2)
    output_ph[after_indx]+=w1/(w1+w2)
    
    before_indx_th=int(angle_th/sec_dis_th)
    if before_indx_th>=18:
        before_indx_th=18-(before_indx_th-17)
    after_indx_th=before_indx_th+1
    if after_indx_th>=18:
        after_indx_th=18-(after_indx_th-16)
    w1_th=abs(angle_th-sec_th[before_indx_th])
    w2_th=abs(angle_th-sec_th[after_indx_th])
    if w2_th>sec_dis_th:
        w2_th=abs(angle_th-(sec_th[after_indx_th]+2*np.pi))
    output_th[before_indx_th]+=w2_th/(w1_th+w2_th)
    output_th[after_indx_th]+=w1_th/(w1_th+w2_th)
    # print before_indx,output[before_indx],after_indx,output[after_indx],angle/np.pi*180
    # raw_input()
    #print(before_indx_th)
    #print(after_indx_th)
    return output_ph, output_th

def get_output0(source):    #!20220630 rename
    sec_center=np.linspace(-np.pi,np.pi,41)
    output=np.zeros(40)
    sec_dis=2*np.pi/40.
    angle=np.arctan2(source[1],source[0])
    before_indx=int((angle+np.pi)/sec_dis)
    after_indx=before_indx+1
    if after_indx>=40:
        after_indx-=40
    w1=abs(angle-sec_center[before_indx])
    w2=abs(angle-sec_center[after_indx])
    if w2>sec_dis:
        w2=abs(angle-(sec_center[after_indx]+2*np.pi))
        #print w2
    output[before_indx]+=w2/(w1+w2)
    output[after_indx]+=w1/(w1+w2)
    # print before_indx,output[before_indx],after_indx,output[after_indx],angle/np.pi*180
    # raw_input()
    return output

def get_output_2source(sources):
    sec_center=np.linspace(-np.pi,np.pi,41)
    output=np.zeros(40)
    sec_dis=2*np.pi/40.
    ws=np.array([np.mean(source["counts"]) for source in sources])
    print('ws_point1')   #!20220303
    ws=ws/ws.sum()
    for i in range(len(sources)):
        source =sources[i]
        angle=np.arctan2(source["position"][1],source["position"][0])
        before_indx=int((angle+np.pi)/sec_dis)
        after_indx=before_indx+1
        if after_indx>=40:
            after_indx-=40
        w1=abs(angle-sec_center[before_indx])
        w2=abs(angle-sec_center[after_indx])
        if w2>sec_dis:
            w2=abs(angle-(sec_center[after_indx]+2*np.pi))
            #print w2
        output[before_indx]+=w2/(w1+w2)*ws[i]
        output[after_indx]+=w1/(w1+w2)*ws[i]
    #print angle,sec_center[before_indx],sec_center[after_indx]
    return output

def get_output_2source_2(sources):
    sec_center=np.linspace(-np.pi,np.pi,41)
    output=np.zeros(40)
    sec_dis=2*np.pi/40.
    ws=np.array([source["intensity"] for source in sources])
    print('ws_point2')   #!20220303
    ws=ws/ws.sum()
    for i in range(len(sources)):
        source =sources[i]
        angle=np.arctan2(source["position"][1],source["position"][0])
        indx=int(round((angle+np.pi)/sec_dis))
        #print indx
        #after_indx=before_indx+1
        if indx>=40:
            indx-=40
        #w1=abs(angle-sec_center[before_indx])
        #w2=abs(angle-sec_center[after_indx])
        #output[before_indx]+=w2/(w1+w2)*ws[i]
        output[indx]+=ws[i]
    #print angle,sec_center[before_indx],sec_center[after_indx]
    return output


# xdata=[]
# ydata=[]
# sources_list=[]
# #yangle=[]
# names=[]

# #path='../../data/drd/data_2source_0302'
path='../../data/drd/data_0402'
path = 'openmc/discrete_data_20220502_v1.1'
path = 'openmc/discrete_data_20220503_v1.1.1'
path = 'openmc/discrete_data_20220630_5^3_v1'    #!20220630
#path='openmc/data_20220301_3.1/' # mean = 1, stdev = 1
#path='openmc/data_20220226_1.2/'
#path='openmc/datafolder_test/'
# files=os.listdir(path)
# for filename in files:
#     #print os.path.join(path,filename)
#     if not filename.endswith('.json'):continue
#     with open(os.path.join(path,filename),'r') as f:
#         data=json.load(f)
#         names.append(filename)
#         xdata.append(data['input'])
#         ydata.append(get_output(data['source']))
#         source=data['source']
#         #yangle.append(np.arctan2(source[1],source[0]))
#         sources_list.append(data['source'])

# xdata=np.array(xdata)
# ydata=np.array(ydata)

# xx=xdata
# yy=ydata

# test_size=600
# training_size=ydata.shape[0]-test_size
# data_size=xx.shape[0]

# # mm=xx[:,:].mean(axis=0,keepdims=True)
# # vv=xx[:,:].var(axis=0,keepdims=True)
# # mm=np.tile(mm,(training_size,1))
# # vv=np.tile(vv,(training_size,1))
# # #mm=mm.reshape((x_data.shape[0],-1))
# # x_data=(xx[0:training_size,:]-mm)/np.sqrt(vv)
# # y_data=yy[0:training_size,:]

# # mm=xx[:,:].mean(axis=0,keepdims=True)
# # vv=xx[:,:].var(axis=0,keepdims=True)
# # mm=np.tile(mm,(test_size,1))
# # vv=np.tile(vv,(test_size,1))
# # x_test=(xx[training_size:training_size+test_size,:]-mm)/np.sqrt(vv)
# # y_test=yy[training_size:training_size+test_size,:]

# mm=xx[:,:].mean(axis=1,keepdims=True)
# vv=xx[:,:].var(axis=1,keepdims=True)
# mm=np.tile(mm,(1,xx.shape[1]))
# vv=np.tile(vv,(1,xx.shape[1]))
# #mm=mm.reshape((x_data.shape[0],-1))
# x_data=(xx[0:training_size,:]-mm[0:training_size,:])/np.sqrt(vv[0:training_size,:])
# y_data=yy[0:training_size,:]

# x_test=(xx[training_size:training_size+test_size,:]-mm[training_size:training_size+test_size,:])/np.sqrt(vv[training_size:training_size+test_size,:])
# y_test=yy[training_size:training_size+test_size,:]

# dataset_size=x_data.shape[0]
# testset_size=y_test.shape[0]
# print x_data.shape,y_data.shape
# print dataset_size,testset_size

# x_size=xx.shape[1]
# y_size=yy.shape[1]

# print x_size,y_size


class Dataset(object):  #!!
    """docstring for Datasedt"""
    def __init__(self, output_fun=get_output,path=path):
        super(Dataset, self).__init__()
        #self.arg = arg
        files=os.listdir(path)
        self.names=[]
        self.source_list=[]
        xdata=[]
        ydata=[]
        ydata_ph = []
        ydata_th = []
        for filename in files:
            #print os.path.join(path,filename)
            if not filename.endswith('.json'):continue
            with open(os.path.join(path,filename),'r') as f:
                data=json.load(f)
                self.names.append(filename)
                source=data['source']   #!20220630
                output_ph, output_th = get_output(source)   #!20220630
                output_mtrx = np.einsum('ij,jk->ik', output_th.reshape(-1, 1), output_ph.reshape(1, -1))    #!20220630
                xdata.append(data['input'])
                ydata.append(output_mtrx)    #!!
                ydata_ph.append(output_ph)  #!20220630
                ydata_th.append(output_th)  #!20220630
                #yangle.append(np.arctan2(source[1],source[0]))
                self.source_list.append(data['source'])
                #print(xdata)    #!20220119
                #print(len(xdata))    #!20220119
                #print('\n')
                #print(filename)

        xdata=np.array(xdata)
        ydata=np.array(ydata)
        ydata_ph=np.array(ydata_ph)
        ydata_th=np.array(ydata_th)

        #xx=xdata
        #yy=ydata

        self.xdata=xdata
        self.ydata=ydata
        self.ydata_ph=ydata_ph  #!20220630
        self.ydata_th=ydata_th  #!20220630
        self.data_size=xdata.shape[0]

class Trainset(object): #!!
    """docstring for Trainset"""
    #def __init__(self, xdata,ydata,info=None,source_num=[2],prob=[1.]):
    def __init__(self, xdata,ydata,ydata_ph, ydata_th, info=None,source_num=[1],prob=[1.]): #!20220630
        super(Trainset, self).__init__()
        #self.arg = arg
        self.info=info
        self.xdata=xdata
        self.ydata=ydata
        self.ydata_ph=ydata_ph  #!20220630
        self.ydata_th=ydata_th  #!20220630
        self.ws=xdata.mean(axis=1)
        self.data_size=xdata.shape[0]
        self.x_size=xdata.shape[1]
        self.y_size=ydata.shape #!20220630
        self.y_ph_size=ydata_ph.shape[1] #!20220630
        self.y_th_size=ydata_th.shape[1]
        self.index_list=np.arange(self.data_size)

        self.source_num=source_num
        self.prob=prob
        #self.data_size=xdata.shape[1]

    def get_batch(self,bs):
        source_num=self.source_num
        prob=self.prob

        xs=[]
        ys=[]
        ys_ph=[]    #!20220630
        ys_th=[]    #!20220630
        for i in range(bs):
            #x,y=self.get_one_data(source_num,prob)
            x,y,y_ph,y_th=self.get_one_data(source_num,prob)  #!20220630
            x=x.reshape(1,-1)
            #y=y.reshape(1,-1)
            y_ph=y_ph.reshape(1,-1) #!20220630
            y_th=y_th.reshape(1,-1) #!20220630
            xs.append(x)
            ys.append(y)
            ys_ph.append(y_ph)    #!20220630
            ys_th.append(y_th)    #!20220630
            pass

        xx=np.concatenate(xs)
        yy=np.concatenate(ys)
        yy_ph=np.concatenate(ys_ph)   #!20220630
        yy_th=np.concatenate(ys_th)   #!20220630

        mm=xx[:,:].mean(axis=1,keepdims=True)
        vv=xx[:,:].var(axis=1,keepdims=True)
        mm=np.tile(mm,(1,xx.shape[1]))
        vv=np.tile(vv,(1,xx.shape[1]))
        #mm=mm.reshape((x_data.shape[0],-1))
        xx=(xx-mm)/np.sqrt(vv)

        #return xx,yy    #!20220630
        return xx,yy,yy_ph,yy_th

    def get_batch_fixsource(self,bs,source_num):
        #source_num=self.source_num
        # prob=self.prob
        # y_size = self.y_size

        # ns = np.random.choice(source_num,p=np.array(prob,dtype=np.float64)/np.sum(prob))

        # section_num = np.array([2*ns, 2*ns-1, 2*ns-2])

        xs=[]
        ys=[]
        ys_ph=[]   #!20220630
        ys_th=[]   #!20220630
        while True:
            if len(xs)>=bs:break
            #x,y=self.get_one_data([source_num],[1.])
            x,y,y_ph,y_th=self.get_one_data([source_num],[1.])  #!20220630
            #if np.where(y!=0)[0].shape[0] != source_num*2: #!20220630 out
            #    continue
            x=x.reshape(1,-1)
            #y=y.reshape(1,-1)  #!20220630
            y_ph=y_ph.reshape(1,-1)   #!20220630
            y_th=y_th.reshape(1,-1)   #!20220630
            xs.append(x)
            ys.append(y)
            ys_ph.append(y_ph)    #!20220630
            ys_th.append(y_th)    #!20220630
            pass

        xx=np.concatenate(xs)
        yy=np.concatenate(ys)
        yy_ph=np.concatenate(ys_ph)   #!20220630
        yy_th=np.concatenate(ys_th)   #!20220630

        mm=xx[:,:].mean(axis=1,keepdims=True)
        vv=xx[:,:].var(axis=1,keepdims=True)
        mm=np.tile(mm,(1,xx.shape[1]))
        vv=np.tile(vv,(1,xx.shape[1]))
        #mm=mm.reshape((x_data.shape[0],-1))
        xx=(xx-mm)/np.sqrt(vv)

        #return xx,yy    #!20220630
        return xx,yy,yy_ph,yy_th


    def get_one_data(self,source_num,prob):
        num=np.random.choice(source_num,p=np.array(prob,dtype=np.float64)/np.sum(prob))
        data_indx=np.random.choice(self.index_list,size=num)

        x=np.zeros(self.x_size)
        y=np.zeros(self.y_size)
        y_ph=np.zeros(self.y_ph_size)   #!20220630
        y_th=np.zeros(self.y_th_size)

        ws=0.
        for indx in data_indx:
            x+=self.xdata[indx,:]   #!20220119
            ws+=self.ws[indx]
            y+=self.ws[indx]*self.ydata[indx,:,:] #!220630
            y_ph+=self.ws[indx]*self.ydata_ph[indx,:] #!220630
            y_th+=self.ws[indx]*self.ydata_th[indx,:] #!220630

        #print('ws_point3')   #!20220303
        #y=y/ws   #!20220303
        if ws !=0:
            y=y/ws   #!20220303
            y_ph=y_ph/ws    #!20220630
            y_th=y_th/ws    #!20220630
        #print('ws_point3')   #!20220303
        #print y.sum()
        #return x,y
        return x,y,y_ph,y_th    #!20220630

    def split(self,split_fold,indx,test_size=None,seed=None):

        source_num=self.source_num
        prob=self.prob

        #sub_size=self.data_size/split_fold
        sub_size=int(self.data_size/split_fold)  #!20220701

        if test_size is None:
            if source_num==[1]:
                test_size=sub_size
            else:
                test_size=sub_size*2

        trains=[]
        # test_x=self.xdata[indx::split_fold,:]
        # test_y=self.ydata[indx::split_fold,:]
        start=indx*sub_size
        end=(indx+1)*sub_size
        if end>self.data_size:
            end=self.data_size

        test_x=self.xdata[start:end,:]
        #test_y=self.ydata[start:end,:,:]
        test_y=self.ydata[start:end,:,:]    #!202206701
        test_y_ph=self.ydata_ph[start:end,:]  #!20220630
        test_y_th=self.ydata_th[start:end,:]  #!20220630
        

        #test=Testset(test_x,test_y,test_size,seed,source_num,prob)
        test=Testset(test_x,test_y,test_y_ph,test_y_th,test_size,seed,source_num,prob)  #!20220701


        train_xs=[]
        train_ys=[]
        train_ys_ph=[] #!20220701
        train_ys_th=[] #!20220701
        for i in range(split_fold):
            if i == indx: continue
            start=i*sub_size
            end=(i+1)*sub_size
            if end>self.data_size:
                end=self.data_size

            train_xs.append(self.xdata[start:end,:])
            #train_ys.append(self.ydata[start:end,:])
            train_ys.append(self.ydata[start:end,:,:])  #!20220701
            train_ys_ph.append(self.ydata_ph[start:end,:])  #!20220701
            train_ys_th.append(self.ydata_th[start:end,:])  #!20220701


        train_x=np.concatenate(train_xs)
        train_y=np.concatenate(train_ys)
        train_y_ph=np.concatenate(train_ys_ph)    #!20220701
        train_y_th=np.concatenate(train_ys_th)    #!20220701

        train=Trainset(train_x,train_y,train_y_ph,train_y_th,source_num=source_num,prob=prob)

        return train,test


class Testset(object):  #!!
    """docstring for Testset"""
    #def __init__(self, xdata,ydata,test_size=None,seed=None,source_num=[2],prob=[1.]):
    def __init__(self, xdata,ydata,ydata_ph,ydata_th,test_size=None,seed=None,source_num=[2],prob=[1.]):
        super(Testset, self).__init__()
        #self.arg = arg
        self.xdata=xdata    # (test_size, a^3)
        self.ydata=ydata    # (test_size, 18, 40)
        self.ydata_ph=ydata_ph  # (test_size, 40)    #!20220630
        self.ydata_th=ydata_th  # (test_size, 18)    #!20220630
        self.ws=xdata.mean(axis=1)
        self.data_size_raw=xdata.shape[0]
        if source_num==[1] or test_size==None:
            self.data_size=self.data_size_raw
        else:
            self.data_size=test_size
        self.x_size=xdata.shape[1]
        self.y_size=ydata.shape  #!20220630
        self.y_ph_size=ydata_ph.shape[1]  #!20220630
        self.y_th_size=ydata_th.shape[1]  #!20220630
        self.index_list=np.arange(self.data_size_raw)



        xx,yy,yy_ph,yy_th=self.gen_data(source_num,prob,seed=seed)  #!20220630
        self.xdata=xx   # (test_size, a^3)
        self.ydata=yy   # (test_size, 18, 40)
        self.ydata_ph=yy_ph # (test_size, 40)   #!20220630
        self.ydata_th=yy_th  # (test_size, 18)   #!20220630

    def gen_data(self,source_num,prob,seed=None):

        np.random.seed(seed)

        xs=[]
        ys=[]
        ys_ph=[]   #!20220630
        ys_th=[]   #!20220630

        if source_num==[1]:# and self.data_size_raw==self.data_size:
            #print 'haha'

            xx=self.xdata   # (test_size, a^3)
            yy=self.ydata   # (test_size, 18, 40)
            yy_ph=self.ydata_ph  # (test_size, 40)  #!20220630
            yy_th=self.ydata_th  # (test_size, 18)  #!20220630

        else:
            for i in range(self.data_size):
                x,y=self.get_one_data(source_num,prob)
                x=x.reshape(1,-1)
                #y=y.reshape(1,-1)   #!20220630
                y_ph=y_ph.reshape(1,-1)   #!20220630
                y_th=y_th.reshape(1,-1)   #!20220630
                xs.append(x)
                ys.append(y)
                ys_ph.append(y_ph)    #!20220630
                ys_th.append(y_th)    #!20220630
                pass

            xx=np.concatenate(xs)
            yy=np.concatenate(ys)
            yy_ph=np.concatenate(ys_ph)   #!20220630
            yy_th=np.concatenate(ys_th)   #!20220630

        mm=xx[:,:].mean(axis=1,keepdims=True)
        vv=xx[:,:].var(axis=1,keepdims=True)
        mm=np.tile(mm,(1,xx.shape[1]))
        vv=np.tile(vv,(1,xx.shape[1]))
        #mm=mm.reshape((x_data.shape[0],-1))
        xx=(xx-mm)/np.sqrt(vv)

        #return xx,yy    #!20220630
        return xx,yy,yy_ph,yy_th    #!20220630


    def get_one_data(self,source_num,prob):
        num=np.random.choice(source_num,p=np.array(prob,dtype=np.float64)/np.sum(prob))
        data_indx=np.random.choice(self.index_list,size=num)

        x=np.zeros(self.x_size)
        y=np.zeros(self.y_size)
        y_ph=np.zeros(self.y_ph_size)   #!20220630
        y_th=np.zeros(self.y_th_size)

        ws=0.
        for indx in data_indx:
            x+=self.xdata[indx,:]
            ws+=self.ws[indx]
            y+=self.ws[indx]*self.ydata[indx,:,:] #!220630
            y_ph+=self.ws[indx]*self.ydata_ph[indx,:] #!220630
            y_th+=self.ws[indx]*self.ydata_th[indx,:] #!220630

        #print('ws_point3')   #!20220303
        #y=y/ws   #!20220303
        if ws !=0:
            y=y/ws   #!20220303
            y_ph=y_ph/ws    #!20220630
            y_th=y_th/ws    #!20220630
        #print('ws_point3')   #!20220303
        #print y.sum()
        #return x,y
        return x,y,y_ph,y_th    #!20220630


    def get_batch(self,bs,indx):
        start=indx*bs
        end=(indx+1)*bs

        if end>self.data_size:
            end=self.data_size
        #return self.xdata[start:end,:],self.ydata[start:end,:] #!20220630
        return self.xdata[start:end,:],self.ydata[start:end,:,:],self.ydata_ph[start:end,:],self.ydata_th[start:end,:]  #!20220630

class FilterData2(object):
    """docstring for FilterData"""
    def __init__(self, filterpath):
        super(FilterData2, self).__init__()
        #self.arg = arg
        self.path=filterpath

        filter_data=[]
        files=os.listdir(filterpath)

        filter_types = ['far', 'near']

        for i in filter_types:
            filter_data.append([])

        for filename in files:
            #print os.path.join(path,filename)
            if not filename.endswith('.json'):continue
            with open(os.path.join(filterpath,filename),'r') as f:
                data=json.load(f)
                #names.append(filename)
                for i,filter_type in enumerate(filter_types):
                    if filter_type in filename:
                        filter_data[i].append(data['input'])    #!!sort??
                #ydata.append(data['output'])
                #source=data['source']
                #yangle.append(np.arctan2(source[1],source[0]))
                #sources.append(data['source'])

        filter_data=np.array(filter_data)
        filter_data = filter_data.reshape((-1,filter_data.shape[-1]))   #!! dimension change!
        # print filter_data.shape

        mm=filter_data[:,:].mean(axis=1,keepdims=True)
        vv=filter_data[:,:].var(axis=1,keepdims=True)
        mm=np.tile(mm,(1,filter_data.shape[1]))
        vv=np.tile(vv,(1,filter_data.shape[1]))
        #mm=mm.reshape((x_data.shape[0],-1))
        filter_data=(filter_data[:,:]-mm[:,:])/np.sqrt(vv[:,:])

        self.data=filter_data
        self.size=filter_data.shape[0]


class FilterData(object):
    """docstring for FilterData"""
    def __init__(self, filterpath):
        super(FilterData, self).__init__()
        #self.arg = arg
        self.path=filterpath

        filter_data=[]
        files=os.listdir(filterpath)

        for filename in files:
            #print os.path.join(path,filename)
            if not filename.endswith('.json'):continue
            with open(os.path.join(filterpath,filename),'r') as f:
                data=json.load(f)
                #names.append(filename)
                filter_data.append(data['input'])
                
                #ydata.append(data['output'])
                #source=data['source']
                #yangle.append(np.arctan2(source[1],source[0]))
                #sources.append(data['source'])
        filter_data=np.array(filter_data)

        mm=filter_data[:,:].mean(axis=1,keepdims=True)
        vv=filter_data[:,:].var(axis=1,keepdims=True)
        mm=np.tile(mm,(1,filter_data.shape[1]))
        vv=np.tile(vv,(1,filter_data.shape[1]))
        #mm=mm.reshape((x_data.shape[0],-1))
        filter_data=(filter_data[:,:]-mm[:,:])/np.sqrt(vv[:,:])

        self.data=filter_data
        self.size=filter_data.shape[0]

filterpath='../../data/drd/filter_0407'
filterpath ='openmc/disc_filter_data_20220630_5^3_v1'    #!20220630
filterdata=FilterData(filterpath)  
filterdata2=FilterData2(filterpath)      

def load_data(test_size,train_size=None,test_size_gen=None,output_fun=get_output,path=path,source_num=[1],prob=[1.],seed=None):


    if test_size_gen is None:
        if source_num==[1]:
            test_size_gen=test_size
        else:
            test_size_gen=test_size*2


    data_set=Dataset(output_fun=get_output,path=path)    #!!
    data_size=data_set.data_size

    if train_size is None:
        train_size=data_set.data_size-test_size

    #train_set=Trainset(data_set.xdata[0:train_size,:],data_set.ydata[0:train_size,:],source_num=source_num,prob=prob)
    train_set=Trainset(data_set.xdata[0:train_size,:],data_set.ydata[0:train_size,:,:],data_set.ydata_ph[0:train_size,:],data_set.ydata_th[0:train_size,:],source_num=source_num,prob=prob)   #!20220701
    test_set=Testset(data_set.xdata[train_size:data_size,:],
            data_set.ydata[train_size:data_size,:,:],data_set.ydata_ph[train_size:data_size,:],data_set.ydata_th[train_size:data_size,:],
            test_size_gen,source_num=source_num,prob=prob,seed=seed)

    return train_set,test_set

# def load_data_1source(test_size,train_size=None,output_fun=get_output,path=path):


#     data_set=Dataset(output_fun=get_output,path=path)
#     data_size=data_set.data_size

#     if train_size is None:
#         train_size=data_set.data_size-test_size

#     train_set=Trainset(data_set.xdata[0:train_size,:],data_set.ydata[0:train_size,:],source_num=[1])
#     test_set=Testset(data_set.xdata[train_size:data_size,:],
#             data_set.ydata[train_size:data_size,:],
#             test_size,source_num=[1])

#     return train_set,test_set

#%%
if __name__ == '__main__':
    # train_set,test_set=load_data(600)
    # x,y=train_set.get_batch(10)
    # x,y=test_set.get_batch(11,54)
    # a,b=train_set.split(5,0)
    # print test_set.data_size
    # print a.get_batch(10)

    train_set,test_set=load_data(100)

    #print(test_set.data_size)#.get_batch(1,10)
    #print(train_set.data_size)#.get_batch(1))

    tt,tt_t=train_set.split(3,1)

    print("finish generating datasets!")
    #print(tt.data_size)
    #print(tt.get_batch_fixsource(6,2))
    # print x.shape,y.shape
    # filterdata=FilterData(filterpath)
    # print filterdata.data[0,:]
    # print filterdata.size
    # np.random.seed(1)
    # print np.random.rand(1)
    # np.random.seed(None)
    # print np.random.rand(1)

# %%
