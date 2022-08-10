#%%
import numpy as np
import json
import os
import torch   #!20220508


def get_output(source, num):    #!20220728
    #sec_center=np.linspace(-np.pi,np.pi,41)
    sec_center=np.linspace(-np.pi,np.pi,num+1)
    output=np.zeros(num)#(40)
    sec_dis=2*np.pi/num #40.
    angle=np.arctan2(source[1],source[0])
    before_indx=int((angle+np.pi)/sec_dis)
    if before_indx>=num: #!20220430 (actually no need to add these two lines..)
        before_indx-=num
    after_indx=before_indx+1
    if after_indx>=num:
        after_indx-=num
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


# def get_output(source):
#     sec_center=np.linspace(-np.pi,np.pi,41)
#     output=np.zeros(40)
#     sec_dis=2*np.pi/40.
#     angle=np.arctan2(source[1],source[0])
#     before_indx=int((angle+np.pi)/sec_dis)
#     after_indx=before_indx+1
#     if after_indx>=40:
#         after_indx-=40
#     w1=abs(angle-sec_center[before_indx])
#     w2=abs(angle-sec_center[after_indx])
#     if w2>sec_dis:
#         w2=abs(angle-(sec_center[after_indx]+2*np.pi))
#         #print w2
#     output[before_indx]+=w2/(w1+w2)
#     output[after_indx]+=w1/(w1+w2)
#     # print before_indx,output[before_indx],after_indx,output[after_indx],angle/np.pi*180
#     # raw_input()
#     return output

def get_output_2source(sources, num):
    sec_center=np.linspace(-np.pi,np.pi,num+1)
    output=np.zeros(num)
    sec_dis=2*np.pi/num
    ws=np.array([np.mean(source["counts"]) for source in sources])
    #print('ws_point1')   #!20220303
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
# path='../../data/drd/data_0402'
# path = 'openmc/discrete_data_20220502_v1.1' #!20220508
# path = 'openmc/discrete_2x2_data_20220627_v1'    #!20220626
#path = 'openmc/discrete_data_20220503_v1.1.1'
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


class Dataset(object):
    """docstring for Datasedt"""
    def __init__(self, seg_angles, output_fun,path):    #!20220729
    #def __init__(self, output_fun=get_output,path=path):
        super(Dataset, self).__init__()
        #self.arg = arg
        files=os.listdir(path)
        self.names=[]
        self.source_list=[]
        xdata=[]
        ydata=[]
        xdata_dict=dict() #!20220508
        ydata_dict=dict() #!20220508
        keys = []   #!20220508
        for filename in files:
            #print os.path.join(path,filename)
            if not filename.endswith('.json'):continue
            with open(os.path.join(path,filename),'r') as f:
                data=json.load(f)
                self.names.append(filename)
                xdata.append(data['input'])
                ydata.append(output_fun(data['source'], seg_angles))
                source=data['source']
                #yangle.append(np.arctan2(source[1],source[0]))
                self.source_list.append(source)
                keys.append(filename)   #!20220508
                xdata_dict[filename]=data['input'] #!20220508
                ydata_dict[filename]=output_fun(data['source'], seg_angles) #!20220508
                #print(xdata)    #!20220119
                #print(len(xdata))    #!20220119
                #print('\n')
                #print(filename)

        xdata=np.array(xdata)
        ydata=np.array(ydata)

        xx=xdata
        yy=ydata
        kk=keys   #!20220508

        self.xdata=xdata
        self.ydata=ydata
        self.xdata_dict=xdata_dict    #!20220508
        self.ydata_dict=ydata_dict
        self.data_size=xdata.shape[0]
        self.keys = keys   #!20220508

class Trainset(object):
    """docstring for Trainset"""
    #def __init__(self, xdata,ydata,info=None,source_num=[2],prob=[1.]):
    #def __init__(self, xdata,ydata,keys, info=None,source_num=[2],prob=[1.]):
    def __init__(self, xdata_dict, ydata_dict, keys, info=None,source_num=[2],prob=[1.]): #!20220508
        super(Trainset, self).__init__()
        #self.arg = arg
        self.info=info
        #self.xdata=xdata #!20220508
        #self.ydata=ydata #!20220508
        self.keys = keys   #!20220508
        self.xdata_dict=xdata_dict #!20220508
        self.ydata_dict=ydata_dict #!20220508
        xdata=[] #!20220508
        ydata=[] #!20220508
        for k in self.keys: #!20220508
            xdata.append(xdata_dict[k])
            ydata.append(xdata_dict[k])
        xdata=np.array(xdata) #!20220508
        ydata=np.array(ydata) #!20220508
        self.xdata=xdata #!20220508
        self.ydata=ydata #!20220508
        self.ws=xdata.mean(axis=1)
        self.data_size=xdata.shape[0]
        self.x_size=xdata.shape[1]
        self.y_size=ydata.shape[1]
        self.index_list=np.arange(self.data_size)

        self.source_num=source_num
        self.prob=prob
        #self.data_size=xdata.shape[1]

    def get_batch_dict(self,bs):
        source_num=self.source_num
        prob=self.prob

        xs=[]
        ys=[]
        for i in range(bs):
            #x,y=self.get_one_data(source_num,prob)
            x,y=self.get_one_data(i,source_num,prob)
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
        #mm=mm.reshape((x_data.shape[0],-1))
        xx=(xx-mm)/np.sqrt(vv)

        xx_dict=dict()
        yy_dict=dict()
        
        #!20220508
        for i in range(len(self.keys)):
            key = self.keys[i]
            xx_line = xx[i, :]
            yy_line = yy[i, :]
            xx_dict[key] = xx_line
            yy_dict[key] = yy_line

        #return xx,yy
        return xx_dict,yy_dict
    

    def get_batch(self,bs):
        source_num=self.source_num
        prob=self.prob

        xs=[]
        ys=[]
        for i in range(bs):
            #x,y=self.get_one_data(source_num,prob)
            x,y=self.get_one_data(i,source_num,prob) #!20220508
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
        #mm=mm.reshape((x_data.shape[0],-1))
        xx=(xx-mm)/np.sqrt(vv)

        return xx,yy



    def get_batch_fixsource(self,bs,source_num):
        #source_num=self.source_num
        # prob=self.prob
        # y_size = self.y_size

        # ns = np.random.choice(source_num,p=np.array(prob,dtype=np.float64)/np.sum(prob))

        # section_num = np.array([2*ns, 2*ns-1, 2*ns-2])

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
        #mm=mm.reshape((x_data.shape[0],-1))
        xx=(xx-mm)/np.sqrt(vv)

        return xx,yy


    #def get_one_data(self,source_num,prob):
    def get_one_data(self,data_indx, source_num,prob): #!20220508
        num=np.random.choice(source_num,p=np.array(prob,dtype=np.float64)/np.sum(prob))
        print('random_choice_source') #!20220508
        print(num) #!20220508
        #data_indx=np.random.choice(self.index_list,size=num)
        #print('random_choice_data_index')
        print('>>> data_index in the correct order: ' + str(data_indx)) #!20220508
        

        x=np.zeros(self.x_size)
        y=np.zeros(self.y_size)

        ws=0.
        #for indx in data_indx:
            #x+=self.xdata[indx,:]   #!20220119
            #ws+=self.ws[indx]
            #y+=self.ws[indx]*self.ydata[indx,:]

        #for indx in data_indx:
        x+=self.xdata[data_indx,:]   #!20220119
        ws+=self.ws[data_indx] #!20220508
        y+=self.ws[data_indx]*self.ydata[data_indx,:] #!20220508

        #print('ws_point3')   #!20220303
        #y=y/ws   #!20220303
        if ws !=0:
            y=y/ws   #!20220303
        #print('ws_point3')   #!20220303
        #print y.sum()
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
        # test_x=self.xdata[indx::split_fold,:]
        # test_y=self.ydata[indx::split_fold,:]
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

        #train=Trainset(train_x,train_y,source_num=source_num,prob=prob)
        train=Trainset(train_x,train_y, source_num=source_num,prob=prob)

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

        if True:    #!20220804
        #if source_num==[1]:# and self.data_size_raw==self.data_size:
            #print 'haha'

            xx=self.xdata
            yy=self.ydata

        else:
            for i in range(self.data_size):
                x,y=self.get_one_data(source_num,prob)
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
        #mm=mm.reshape((x_data.shape[0],-1))
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

        #print('ws_point4')   #!20220303
        #y=y/ws  #!20220305
        if ws != 0:
            y=y/ws  #!20220305
        #print y.sum()
        return x,y

    def get_batch(self,bs,indx):
        start=indx*bs
        end=(indx+1)*bs

        if end>self.data_size:
            end=self.data_size
        return self.xdata[start:end,:],self.ydata[start:end,:]

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
                        filter_data[i].append(data['input'])
                #ydata.append(data['output'])
                #source=data['source']
                #yangle.append(np.arctan2(source[1],source[0]))
                #sources.append(data['source'])

        filter_data=np.array(filter_data)
        filter_data = filter_data.reshape((-1,filter_data.shape[-1]))
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

# filterpath='../../data/drd/filter_0407'
# filterpath ='openmc/disc_filter_2x2_data_20220627_v1.1' #!20220626
# #filterdata=FilterData(filterpath)  
# filterdata2=FilterData2(filterpath)      

def load_data(test_size,train_size,test_size_gen,seg_angles, output_fun,path,source_num,prob,seed):


    if test_size_gen is None:
        if source_num==[1]:
            test_size_gen=test_size
        else:
            test_size_gen=test_size*2


    data_set=Dataset(seg_angles, output_fun=output_fun,path=path)
    data_size=data_set.data_size

    if train_size is None:
        train_size=data_set.data_size-test_size

    #train_set=Trainset(data_set.xdata[0:train_size,:],data_set.ydata[0:train_size,:], keys, source_num=source_num,prob=prob)
    train_set=Trainset(data_set.xdata_dict,data_set.ydata_dict, data_set.keys, source_num=source_num,prob=prob) #!20220508
    test_set=Testset(data_set.xdata[train_size:data_size,:],
            data_set.ydata[train_size:data_size,:],
            test_size_gen,source_num=source_num,prob=prob,seed=seed)

    #return train_set,test_set #!20220508

    return train_set,test_set, data_set #!20220508

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
if __name__ == '__main__': #!20220508
    path = 'openmc/discrete_5x5_2src_128_data_20220809_v1.1'  #!20220716
    filterpath ='openmc/disc_filter_5x5_128_data_20220809_v1.1'    #!20220716
    filterdata=FilterData(filterpath)  
    filterdata2=FilterData2(filterpath)      
    # filterdata=FilterData(filterpath)  
    # filterdata2=FilterData2(filterpath)      
    seg_angles = 128
    GPU_INDEX = 1#0
    USE_CPU = False
    # print torch.cuda.is_available()
    if torch.cuda.is_available() and not USE_CPU:
        DEFAULT_DEVICE = torch.device("cuda:%d"%GPU_INDEX) 
        torch.cuda.set_device(GPU_INDEX)
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    else:
        DEFAULT_DEVICE = torch.device("cpu")

    DEFAULT_DTYPE = torch.double

    #num_data = 3
    #batch_size = 1000
    datas = []
    #train_set,test_set, data_set=load_data(test_size=0,train_size=None,test_size_gen=None,seg_angles=seg_angles,output_fun=get_output,path=path,source_num=[1],prob=[1.],seed=None)
    train_set,test_set, data_set=load_data(test_size=0,train_size=None,test_size_gen=None,seg_angles=seg_angles,
                                           output_fun=get_output_2source,path=path,source_num=[1, 1],prob=[1., 1.],seed=None)  #!20220804
    #train_set,test_set=load_data(0, source_num=[1])
    print('train_len')
    train_len = train_set.data_size
    print(train_len)
    batch_size = train_len
    #for i in range(num_data):
    data_x, data_y=train_set.get_batch(batch_size)
    print('x_shape')
    print(data_x.shape)
    print('y_shape')
    print(data_y.shape)
            
    data_x = torch.from_numpy(data_x).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)    #!20211230
    data_y = torch.from_numpy(data_y).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)    #!20211230

    datas.append((data_x,data_y))

    #=======The dimension check!=========#!20220508
    print("=======The dimension check start!=========")
    print("datay")
    print(data_y)
    print(data_y.shape)
    print(torch.sum(data_y))#,dim=1,keepdim=True))
    #=======The dimension check!=========#!20220508

    #=======The dimension check!=========#!20220508
    print("=======The dimension check start!=========")
    print("datay")
    key_list = data_set.keys
    y_num = data_y.shape[0]
    ng_list = []
    old_ng_list = []
    for i in range(y_num):
        k = key_list[i]
        if k.endswith('.json'):   ##!20220720
            y_line = data_y[i,:]
            y_line_sum = float(torch.sum(y_line))
            print(k + ": "+ str(y_line_sum))
            if y_line_sum == 0:
                ng_list.append(k)
        else:   #!20220720
            old_ng_list.append(k)
        #print(y_line)
    #print(data_y)
    print(data_y.shape)
    print(torch.sum(data_y))#,dim=1,keepdim=True))
    #=======The dimension check!=========#!20220508
    
    #+++++++++++++++
    
    data_x_dict, data_y_dict=train_set.get_batch_dict(batch_size)
    print('x_len')
    print(len(data_x_dict))
    print('y_len')
    print(len(data_y_dict))
            
    #data_x = torch.from_numpy(data_x).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)    #!20211230
    #data_y = torch.from_numpy(data_y).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)    #!20211230

    #datas.append((data_x,data_y))

    #=======The dimension check!=========#!20220508
    print("=======The dimension check start!=========")
    print("datay")
    print(data_y_dict)
    print(len(data_y_dict))
    #print(torch.sum(data_y))#,dim=1,keepdim=True))
    #=======The dimension check!=========#!20220508

    #=======The dimension check!=========#!20220508
    print("=======The dimension check start (dict)!=========")
    print("datay")
    key_list = data_set.keys
    y_dict_num = len(data_y_dict)
    ng_list_2 = []
    for i in range(y_dict_num):
        k = key_list[i]
        y_dict_line = data_y_dict[k]
        y_dict_line_sum = sum(y_dict_line)
        print(k + ": "+ str(y_dict_line_sum))
        if y_dict_line_sum == 0:
            ng_list_2.append(k)
        #print(y_line)
    #print(data_y)
    #print(data_y.shape)
    #print(torch.sum(data_y))#,dim=1,keepdim=True))
    #=======The dimension check!=========#!20220508
    
    os.system(f"cp -r {path} {path}_original") #!20220626
    
    for fn in ng_list:
        os.system(f"mv {path}/{fn} {path}/{fn[:-4]}txt")
    print("old_ng_list length: ", len(old_ng_list))
    print("ng_list length: ", len(ng_list))
    total_ng_list = list(set(ng_list + old_ng_list))    #!20220720
    print("total_ng_list length: ", len(total_ng_list))

                
# %%
