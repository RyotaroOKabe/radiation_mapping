import torch
import numpy as np

def sinkhorn_step(a,b,K,Kp,u,v,cpt,err):
    KtransposeU=torch.einsum('aji,aj->ai',K,u)
    v = torch.div(b, KtransposeU)
    u=1. / torch.einsum('aij,aj->ai',Kp,v)#*a
    if cpt % 10 == 0:
        tmp=torch.einsum('ai,aij,aj->aj', u, K, v)
    cpt = cpt + 1
    return a,b,K,Kp,u,v,cpt,err

def emd_pop_zero_batch(a,M,n=0):
    batch_size=a.shape[0]
    none_zero_indx=np.where(a!=0)

    batch_M=np.tile(np.reshape(M,(-1,M.shape[0],M.shape[-1])),[batch_size,1,1])
    new_a=a[none_zero_indx].reshape(batch_size,-1)
    new_M=batch_M[none_zero_indx[0],none_zero_indx[1],:].reshape(batch_size,-1,M.shape[1])
    return new_a,new_M

def sinkhorn_torch(a, b, M, reg, numItermax=1000, stopThr=1e-9):
    batch_size=a.shape[0]
    Nini = a.shape[1]
    Nfin = b.shape[1]
    u = torch.ones([1,Nini]).to(a) / Nini
    u = u.repeat(batch_size,1)
    v = torch.ones([1,Nfin]).to(a) / Nfin
    v = v.repeat(batch_size,1)

    batch_M=M
    batch_K=torch.exp(-batch_M/reg)

    one_devide_a=torch.reshape(1./a,(batch_size,a.shape[1],-1)).repeat(1,1,batch_K.shape[2])
    Kp=one_devide_a*batch_K

    cpt = 0
    err = 1.


    for sb in range(numItermax):
        a,b,batch_K,Kp,u,v,cpt,err=sinkhorn_step(a,b,batch_K,Kp,u,v,cpt,err)

    res= torch.einsum('ai,aij,aj,aij->a', u, batch_K, v, batch_M)

    return torch.sum(res)

if __name__ == '__main__':
    from pyemd import emd
    import ot
    data1=np.array([[0.00001,0.000001,0.00001,0.0000001,1,0.00000001,0.0000001,1,0.0000001,0.000000001],
        [1,1,0.00001,0.0000001,1,0.00000001,0.0000001,1,0.0000001,0.000000001]
        ],dtype=np.float64).reshape(2,-1)
    data2=np.array([[0,1,0,0,1,0,2,0,1,0],[0,1,0,0,1,0,2,0,1,0]],dtype=np.float64).reshape(2,-1)

    data1 = data1/np.sum(data1,axis=1,keepdims=True)
    data2 = data2/np.sum(data2,axis=1,keepdims=True)
    n = data2.shape[1]
    M=np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            M[i,j]=min(abs(i-j),j+n-i,i+n-j)**2
    data2_new,M_new=emd_pop_zero_batch(data2,M)

    print(sinkhorn_torch(torch.as_tensor(data2_new),torch.as_tensor(data1),torch.as_tensor(M_new),reg=5., numItermax=100).item())
    print(ot.sinkhorn2(data2[1,:],data1[1,:],M,5)+ot.sinkhorn2(data2[0,:],data1[0,:],M,5))
