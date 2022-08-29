#%%
import numpy as np
import json
import os

def get_output(source):
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
# path='../../data/drd/data_0402'
# path = 'openmc/discrete_data_20220502_v1.1'
# path = 'openmc/discrete_data_20220503_v1.1.1'
# path = 'openmc/discrete_2x2_data_20220627_v1'    #!20220626
# path = 'openmc/discrete_tetris_T_data_20220716_v1'  #!20220716
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
    #def __init__(self, output_fun=get_output,path=path):
    def __init__(self, output_fun,path):
        super(Dataset, self).__init__()
        #self.arg = arg
        files=os.listdir(path)
        self.names=[]
        self.source_list=[]
        xdata=[]
        ydata=[]
        for filename in files:
            #print os.path.join(path,filename)
            if not filename.endswith('.json'):continue
            with open(os.path.join(path,filename),'r') as f:
                data=json.load(f)
                self.names.append(filename)
                xdata.append(data['input'])
                ydata.append(output_fun(data['source']))
                source=data['source']
                #yangle.append(np.arctan2(source[1],source[0]))
                self.source_list.append(data['source'])
                #print(xdata)    #!20220119
                #print(len(xdata))    #!20220119
                #print('\n')
                #print(filename)

        xdata=np.array(xdata)
        ydata=np.array(ydata)

        xx=xdata
        yy=ydata

        self.xdata=xdata
        self.ydata=ydata
        self.data_size=xdata.shape[0]

class Trainset(object):
    """docstring for Trainset"""
    def __init__(self, xdata,ydata,info=None,source_num=[2],prob=[1.]):
        super(Trainset, self).__init__()
        #self.arg = arg
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
        #self.data_size=xdata.shape[1]

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


    def get_one_data(self,source_num,prob):
        num=np.random.choice(source_num,p=np.array(prob,dtype=np.float64)/np.sum(prob))
        data_indx=np.random.choice(self.index_list,size=num)

        x=np.zeros(self.x_size)
        y=np.zeros(self.y_size)

        ws=0.
        for indx in data_indx:
            x+=self.xdata[indx,:]   #!20220119
            ws+=self.ws[indx]
            y+=self.ws[indx]*self.ydata[indx,:]

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

        if source_num==[1]:# and self.data_size_raw==self.data_size:
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
# filterpath ='openmc/disc_filter_2x2_data_20220627_v1.1'
# filterpath ='openmc/disc_filter_tetris_T_data_20220716_v3.1'    #!20220716
# filterdata=FilterData(filterpath)  
# filterdata2=FilterData2(filterpath)      

def load_data(test_size,train_size,test_size_gen,output_fun,path,source_num,prob,seed):


    if test_size_gen is None:
        if source_num==[1]:
            test_size_gen=test_size
        else:
            test_size_gen=test_size*2


    data_set=Dataset(output_fun=get_output,path=path)
    data_size=data_set.data_size

    if train_size is None:
        train_size=data_set.data_size-test_size

    train_set=Trainset(data_set.xdata[0:train_size,:],data_set.ydata[0:train_size,:],source_num=source_num,prob=prob)
    test_set=Testset(data_set.xdata[train_size:data_size,:],
            data_set.ydata[train_size:data_size,:],
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
    path = 'openmc/discrete_tetris_T_data_20220716_v1'  #!20220716
    filterpath ='openmc/disc_filter_tetris_T_data_20220716_v3.1'    #!20220716
    filterdata=FilterData(filterpath)  
    filterdata2=FilterData2(filterpath)      
    # train_set,test_set=load_data(600)
    # x,y=train_set.get_batch(10)
    # x,y=test_set.get_batch(11,54)
    # a,b=train_set.split(5,0)
    # print test_set.data_size
    # print a.get_batch(10)

    #train_set,test_set=load_data(600)
    train_set,test_set=load_data(test_size=50,train_size=None,test_size_gen=None,output_fun=get_output,path=path,source_num=[1],prob=[1.],seed=None)    #!20220716
    #print(test_set.data_size)#.get_batch(1,10)
    #print(train_set.data_size)#.get_batch(1))

    ##tt,tt_t=train_set.split(3,1)

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
