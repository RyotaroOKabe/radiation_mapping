import torch
import numpy as np

def ecdf(p):
    """Estimate the cumulative distribution function.
    The e.c.d.f. (empirical cumulative distribution function) F_n is a step
    function with jump 1/n at each observation (possibly with multiple jumps
    at one place if there are ties).
    For observations x= (x_1, x_2, ... x_n), F_n is the fraction of
    observations less or equal to t, i.e.,
    F_n(t) = #{x_i <= t} / n = 1/n \sum^{N}_{i=1} Indicator(x_i <= t).
    Args:
      p: a 2-D `Tensor` of observations of shape [batch_size, num_classes].
        Classes are assumed to be ordered.
    Returns:
      A 2-D `Tensor` of estimated ECDFs.
    """
    n = p.shape[1]
    triang = torch.t(torch.tril(torch.ones([n,n]).to(p)))
    return torch.matmul(p,triang)


def shift_tf(p,dis=0):
    p1=p[:,0:dis]
    p2=p[:,dis:]
    p=tf.concat([p2,p1],1)
    return p

def shift(p,dis=1):  #!20211230
    if dis==0:
        p=p
    else:
        p1=p[:,0:dis]
        p2=p[:,dis:]
        p=torch.cat([p2,p1],1)
    return p


def emd_loss(p, p_hat, r=2):
    """Compute the Earth Mover's Distance loss.
    Hou, Le, Chen-Ping Yu, and Dimitris Samaras. "Squared Earth Mover's
    Distance-based Loss for Training Deep Neural Networks." arXiv preprint
    arXiv:1611.05916 (2016).
    Args:
      p: a 2-D `Tensor` of the ground truth probability mass functions.
      p_hat: a 2-D `Tensor` of the estimated p.m.f.-s
      r: a constant for the r-norm.
      scope: optional name scope.
    `p` and `p_hat` are assumed to have equal mass as \sum^{N}_{i=1} p_i =
    \sum^{N}_{i=1} p_hat_i
    Returns:
      A 0-D `Tensor` of r-normed EMD loss.
    """
    p=p/torch.sum(p,dim=1,keepdims=True)
    ecdf_p = ecdf(p)
    ecdf_p_hat = ecdf(p_hat)
    emd = torch.mean(torch.pow(torch.abs(ecdf_p - ecdf_p_hat), r), dim= 1)
    emd = torch.pow(emd, 1./r)
    return torch.mean(emd)


def emd_loss_ring(p,p_hat,r=2, small=1e-5):
    """Compute the Earth Mover's Distance loss.
    Hou, Le, Chen-Ping Yu, and Dimitris Samaras. "Squared Earth Mover's
    Distance-based Loss for Training Deep Neural Networks." arXiv preprint
    arXiv:1611.05916 (2016).
    Args:
      p: a 2-D `Tensor` of the ground truth probability mass functions.
      p_hat: a 2-D `Tensor` of the estimated p.m.f.-s
      r: a constant for the r-norm.
      scope: optional name scope.
    `p` and `p_hat` are assumed to have equal mass as \sum^{N}_{i=1} p_i =
    \sum^{N}_{i=1} p_hat_i
    Returns:
      A 0-D `Tensor` of r-normed EMD loss.
    """
    p=p/(torch.sum(p,dim=1,keepdim=True)+small)
    p_hat=p_hat/(torch.sum(p_hat,dim=1,keepdim=True)+small)
    n = p.shape[1]
    ecdf_ps=[]
    ecdf_p_hats=[]
    for i in range(n):
        pp=shift(p,i)
        pp_hat=shift(p_hat,i)
        ecdf_p = ecdf(pp)
        ecdf_p_hat = ecdf(pp_hat)
        ecdf_ps.append(ecdf_p)
        ecdf_p_hats.append(ecdf_p_hat)
    ecdf_p = torch.stack(ecdf_ps)
    ecdf_p_hat=torch.stack(ecdf_p_hats)
    emd = torch.sum(torch.pow(torch.abs(ecdf_p - ecdf_p_hat), r), dim=2)
    emd = torch.min(emd,dim=0)[0]
    emd = torch.pow(emd, 1. / r)
    emd=torch.mean(emd,dim=0)
    return emd

#%%
if __name__ == '__main__':
    from pyemd import emd
    data1=np.array([[0,0,1,0,0,0.1,0,0,0],[0,1,0,0,1,1,0,1,0]],dtype=np.float64).reshape(2,-1)
    data2=np.array([[0,1,0,1,0,0,0,1,0],[0,1,0,0,0,1,0,0,0]],dtype=np.float64).reshape(2,-1)
    a=data1[0].copy().reshape(1,-1)
    b=data2[0].copy().reshape(1,-1)
    print(emd_loss_ring(torch.from_numpy(data1),torch.from_numpy(data2), r=2))

    n=a.shape[1]
    M1=np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            M1[i,j]=min(abs(i-j),j+n-i,i+n-j)#**2

# %%
