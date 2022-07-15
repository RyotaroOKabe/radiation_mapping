#import tensorflow as tf
import torch
import numpy as np
import tensorflow as tf #!20211230

def tril_indices(n, k=0):
    """Return the indices for the lower-triangle of an (n, m) array.
    Works similarly to `np.tril_indices`
    Args:
      n: the row dimension of the arrays for which the returned indices will
        be valid.
      k: optional diagonal offset (see `np.tril` for details).
    Returns:
      inds: The indices for the triangle. The returned tuple contains two arrays,
        each with the indices along one dimension of the array.
    """
    m1 = tf.tile(tf.expand_dims(tf.range(n), axis=0), [n, 1])
    m2 = tf.tile(tf.expand_dims(tf.range(n), axis=1), [1, n])
    mask = (m1 - m2) >= -k
    ix1 = tf.boolean_mask(m2, tf.transpose(mask))
    ix2 = tf.boolean_mask(m1, tf.transpose(mask))
    return ix1, ix2

def ecdf_tf(p):
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
    n = p.get_shape().as_list()[1]
    indices = tril_indices(n)
    indices = tf.transpose(tf.stack([indices[1], indices[0]]))
    ones = tf.ones([n * (n + 1) / 2])
    triang = tf.scatter_nd(indices, ones, [n, n])
    return tf.matmul(p, triang)

def ecdf(p):
    n = p.shape[1]

    #triang = torch.tril(torch.ones([n,n]).to(p)).T #!20211230
    triang = torch.t(torch.tril(torch.ones([n,n]).to(p))) #!20211230
    return torch.matmul(p,triang)


def shift_tf(p,dis=0): #!20211230
    p1=p[:,0:dis]
    p2=p[:,dis:]
    p=tf.concat([p2,p1],1)
    return p
  
#%%
#def shift(p,dis=0): #!20211230
#    p1=p[:,0:dis]
#    p2=p[:,dis:]
#    p=torch.cat([p2,p1],1)
#    return p
  
def shift(p,dis=1):  #!20211230
    if dis==0:
        p=p
    else:
        p1=p[:,0:dis]
        p2=p[:,dis:]
        p=torch.cat([p2,p1],1)
    return p
#%%
def emd_loss_tf(p, p_hat, r=2, scope=None):
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
    p=p/tf.reduce_sum(p,axis=1,keepdims=True)
    with tf.name_scope(scope, 'EmdLoss', [p, p_hat]):
        #p=shift(p,1)
        ecdf_p = ecdf_tf(p)
        ecdf_p_hat = ecdf_tf(p_hat)
        emd = tf.reduce_mean(tf.pow(tf.abs(ecdf_p - ecdf_p_hat), r), axis=-1)
        emd = tf.pow(emd, 1. / r)
        return tf.reduce_mean(emd)

def emd_loss(p, p_hat, r=2):
    p=p/torch.sum(p,dim=1,keepdims=True)
    ecdf_p = ecdf(p)
    ecdf_p_hat = ecdf(p_hat)
    emd = torch.mean(torch.pow(torch.abs(ecdf_p - ecdf_p_hat), r), dim= 1)
    emd = torch.pow(emd, 1./r)
    return torch.mean(emd)

def emd_loss_ring_tf(p, p_hat, r=2, scope=None):
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
    p=p/tf.reduce_sum(p,axis=1,keepdims=True)
    p_hat=p_hat/tf.reduce_sum(p_hat,axis=1,keepdims=True)
    with tf.name_scope(scope, 'EmdLoss', [p, p_hat]):
        #p=shift(p,1)
        n = p.get_shape().as_list()[1]
        ecdf_ps=[]
        ecdf_p_hats=[]
        for i in range(n):
            pp=shift_tf(p,i)
            pp_hat=shift_tf(p_hat,i)
            ecdf_p = ecdf_tf(pp)
            ecdf_p_hat = ecdf_tf(pp_hat)
            ecdf_ps.append(ecdf_p)
            ecdf_p_hats.append(ecdf_p_hat)
        ecdf_p = tf.stack(ecdf_ps)
        ecdf_p_hat=tf.stack(ecdf_p_hats)
        #print(ecdf_p.get_shape())
        emd = tf.reduce_sum(tf.pow(tf.abs(ecdf_p - ecdf_p_hat), r), axis=2)
        emd = tf.reduce_min(emd,axis=0)
        emd = tf.pow(emd, 1. / r)
        emd=tf.reduce_mean(emd,axis=0)
        return emd

def emd_loss_ring(p,p_hat,r=2):
    #p=p/torch.sum(p,dim=1,keepdims=True) #!20211229
    #p_hat=p_hat/torch.sum(p_hat,dim=1,keepdims=True)
    #print("======test0=====") #!20220330
    #print(p_hat)  #!20220303 >> check!
    p_hat_m=torch.mean(p_hat) #,dim=0) #!20220330 testing
    #print(p_hat_m) #!20220330 testing >> This part was no problem
    #p=p/torch.sum(p)  #,dim=1,keepdim=True) #!20211229
    p=p/torch.sum(p,dim=1,keepdim=True) #!20220714
    #print(p)
    p_m=torch.mean(p) #,dim=0) #!20220330 testing
    #print(p_m) #!20220330 testing >> Include nan
    p_hat=p_hat/torch.sum(p_hat,dim=1,keepdim=True) #!20211229
    #print(p_hat)  #!20220303 >> check!
    p_hat_m=torch.mean(p_hat) #,dim=0) #!20220330 testing
    #print(p_hat_m) #!20220330 testing
        #p=shift(p,1)
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
    #print("======test1=====") #!20220330
    #print(ecdf_p)  #!20220303 >> error!
    ecdf_p_m=torch.mean(ecdf_p,dim=0) #!20220330 testing
    #print(ecdf_p_m) #!20220330 testing >> Include nan
    #print("======test2=====") #!20220330
    ecdf_p_hat=torch.stack(ecdf_p_hats)
    #print(ecdf_p_hat)  #!20220303  >> check!!
    ecdf_p_hat_m=torch.mean(ecdf_p_hat,dim=0) #!20220330 testing
    #print(ecdf_p_hat_m) #!20220330 testing >> Include nan
    #print(ecdf_p.get_shape())
    #print("======test3=====") #!20220330
    emd = torch.sum(torch.pow(torch.abs(ecdf_p - ecdf_p_hat), r), dim=2)
    #print(emd)  #!20220303
    emd_m=torch.mean(emd,dim=0) #!20220330 testing
    #print(emd_m) #!20220330 testing  >> Include nan
    #print("======test4=====") #!20220330
    emd = torch.min(emd,dim=0)[0]
    #print(emd)
    emd_m=torch.mean(emd,dim=0) #!20220330 testing
    #print(emd_m) #!20220330 testing >> Include nan
    #print("======test5=====") #!20220330
    #print(emd)  #!20220303 testing
    emd = torch.pow(emd, 1. / r)
    #print("====emd start===")
    #print(emd)
    #print(emd.shape)
    #print(type(emd))
    emd_m=torch.mean(emd,dim=0) #!20220330 testing
    #print(emd_m) #!20220330 testing >> Include nan
    #print(emd)  #!20220303 testing
    #print("======test6=====") #!20220330
    #print("====emd end===")
    emd=torch.mean(emd,dim=0)
    #print(emd)  #!20220303
    emd_m=torch.mean(emd,dim=0) #!20220330 testing
    #print(emd_m) #!20220330 testing >> Include nan
    return emd

# def emd_loss_ring(p,p_hat,r=2):
#     #p=p/torch.sum(p,dim=1,keepdims=True) #!20211229
#     #p_hat=p_hat/torch.sum(p_hat,dim=1,keepdims=True)
#     print("======test0=====") #!20220330
#     print(p_hat)  #!20220303 >> check!
#     p_hat_m=torch.mean(p_hat) #,dim=0) #!20220330 testing
#     print(p_hat_m) #!20220330 testing >> This part was no problem
#     p=p/torch.sum(p)  #,dim=1,keepdim=True) #!20211229
#     print(p)
#     p_m=torch.mean(p) #,dim=0) #!20220330 testing
#     print(p_m) #!20220330 testing >> Include nan
#     p_hat=p_hat/torch.sum(p_hat,dim=1,keepdim=True) #!20211229
#     print(p_hat)  #!20220303 >> check!
#     p_hat_m=torch.mean(p_hat) #,dim=0) #!20220330 testing
#     print(p_hat_m) #!20220330 testing
#         #p=shift(p,1)
#     n = p.shape[1]
#     ecdf_ps=[]
#     ecdf_p_hats=[]
#     for i in range(n):
#         pp=shift(p,i)
#         pp_hat=shift(p_hat,i)
#         ecdf_p = ecdf(pp)
#         ecdf_p_hat = ecdf(pp_hat)
#         ecdf_ps.append(ecdf_p)
#         ecdf_p_hats.append(ecdf_p_hat)
#     ecdf_p = torch.stack(ecdf_ps)
#     print("======test1=====") #!20220330
#     print(ecdf_p)  #!20220303 >> error!
#     ecdf_p_m=torch.mean(ecdf_p,dim=0) #!20220330 testing
#     print(ecdf_p_m) #!20220330 testing >> Include nan
#     print("======test2=====") #!20220330
#     ecdf_p_hat=torch.stack(ecdf_p_hats)
#     print(ecdf_p_hat)  #!20220303  >> check!!
#     ecdf_p_hat_m=torch.mean(ecdf_p_hat,dim=0) #!20220330 testing
#     print(ecdf_p_hat_m) #!20220330 testing >> Include nan
#     #print(ecdf_p.get_shape())
#     print("======test3=====") #!20220330
#     emd = torch.sum(torch.pow(torch.abs(ecdf_p - ecdf_p_hat), r), dim=2)
#     print(emd)  #!20220303
#     emd_m=torch.mean(emd,dim=0) #!20220330 testing
#     print(emd_m) #!20220330 testing  >> Include nan
#     print("======test4=====") #!20220330
#     emd = torch.min(emd,dim=0)[0]
#     print(emd)
#     emd_m=torch.mean(emd,dim=0) #!20220330 testing
#     print(emd_m) #!20220330 testing >> Include nan
#     print("======test5=====") #!20220330
#     #print(emd)  #!20220303 testing
#     emd = torch.pow(emd, 1. / r)
#     #print("====emd start===")
#     print(emd)
#     print(emd.shape)
#     print(type(emd))
#     emd_m=torch.mean(emd,dim=0) #!20220330 testing
#     print(emd_m) #!20220330 testing >> Include nan
#     #print(emd)  #!20220303 testing
#     print("======test6=====") #!20220330
#     #print("====emd end===")
#     emd=torch.mean(emd,dim=0)
#     print(emd)  #!20220303
#     emd_m=torch.mean(emd,dim=0) #!20220330 testing
#     print(emd_m) #!20220330 testing >> Include nan
#     return emd
  
#%%
def ecdf_3d(p):
    n = p.shape[1]

    #triang = torch.tril(torch.ones([n,n]).to(p)).T #!20211230
    triang = torch.t(torch.tril(torch.ones([n,n]).to(p))) #!20211230
    #return torch.matmul(p,triang)
    return torch.einsum('ijk,jl->ilk', p,triang)   #!20220704

def shift_3d(p,dis=1):  #!20211230
    if dis==0:
        p=p
    else:
        p1=p[:,0:dis,:] #!20220704
        p2=p[:,dis:,:]  #!20220704
        p=torch.cat([p2,p1],1)
    return p

def emd_loss_ring_3d(p,p_hat,r=2):
    #p=p/torch.sum(p,dim=1,keepdims=True) #!20211229
    #p_hat=p_hat/torch.sum(p_hat,dim=1,keepdims=True)
    #print("======test0=====") #!20220330
    #print(p_hat)  #!20220303 >> check!
    #p_hat_m=torch.mean(p_hat) #,dim=0) #!20220330 testing
    #print(p_hat_m) #!20220330 testing >> This part was no problem
    #if torch.sum(p) > 0:
    #?if torch.sum(p) > 1:
        #?p=p/torch.sum(p)  #,dim=1,keepdim=True) #!20211229
    p_sum12 = torch.sum(torch.sum(p,dim=1,keepdim=True), dim=2, keepdim=True)   #!20220707
    p=p/p_sum12
    #?p=p/torch.sum(p,dim=1,keepdim=True) #!20220704
    #print(p)
    #p_m=torch.mean(p) #,dim=0) #!20220330 testing
    #print(p_m) #!20220330 testing >> Include nan
    #if torch.sum(p_hat) >0
    #?if torch.sum(p_hat) >1:
        #?p_hat=p_hat/torch.sum(p_hat) #!20220704
    p_hat_sum12 = torch.sum(torch.sum(p_hat,dim=1,keepdim=True), dim=2, keepdim=True)   #!20220707
    p_hat=p/p_hat_sum12
    #?p_hat=p_hat/torch.sum(p_hat,dim=1,keepdim=True) #!20211229
    #print(p_hat)  #!20220303 >> check!
    #p_hat_m=torch.mean(p_hat) #,dim=0) #!20220330 testing
    #print(p_hat_m) #!20220330 testing
        #p=shift(p,1)
    n = p.shape[1]
    ecdf_ps=[]
    ecdf_p_hats=[]
    for i in range(n):
        pp=shift_3d(p,i)
        pp_hat=shift_3d(p_hat,i)
        ecdf_p = ecdf_3d(pp)
        ecdf_p_hat = ecdf_3d(pp_hat)
        ecdf_ps.append(ecdf_p)
        ecdf_p_hats.append(ecdf_p_hat)
    ecdf_p = torch.stack(ecdf_ps)
    #print("======test1=====") #!20220330
    #print(ecdf_p)  #!20220303 >> error!
    #ecdf_p_m=torch.mean(ecdf_p,dim=0) #!20220330 testing
    #print(ecdf_p_m) #!20220330 testing >> Include nan
    #print("======test2=====") #!20220330
    ecdf_p_hat=torch.stack(ecdf_p_hats)
    #print(ecdf_p_hat)  #!20220303  >> check!!
    #ecdf_p_hat_m=torch.mean(ecdf_p_hat,dim=0) #!20220330 testing
    #print(ecdf_p_hat_m) #!20220330 testing >> Include nan
    #print(ecdf_p.get_shape())
    #print("======test3=====") #!20220330
    emd = torch.sum(torch.pow(torch.abs(ecdf_p - ecdf_p_hat), r), dim=2)  #!!
    #print(emd)  #!20220303
    #emd_m=torch.mean(emd,dim=0) #!20220330 testing
    #print(emd_m) #!20220330 testing  >> Include nan
    #print("======test4=====") #!20220330
    emd = torch.min(emd,dim=0)[0]
    #print(emd)
    #emd_m=torch.mean(emd,dim=0) #!20220330 testing
    #print(emd_m) #!20220330 testing >> Include nan
    #print("======test5=====") #!20220330
    #print(emd)  #!20220303 testing
    emd = torch.pow(emd, 1. / r)
    #print("====emd start===")
    #print(emd)
    #print(emd.shape)
    #print(type(emd))
    #emd_m=torch.mean(emd,dim=0) #!20220330 testing
    #print(emd_m) #!20220330 testing >> Include nan
    #print(emd)  #!20220303 testing
    #print("======test6=====") #!20220330
    #print("====emd end===")
    emd=torch.mean(emd,dim=0)
    #print(emd)  #!20220303
    #emd_m=torch.mean(emd,dim=0) #!20220330 testing
    #print(emd_m) #!20220330 testing >> Include nan
    #return emd
    return torch.sum(emd)   ##!20220704

#%%

if __name__ == '__main__':
  from pyemd import emd
  data1 = torch.randn(256, 40)
  pred1 = torch.randn(256, 40)
  
  data2 = torch.randn(256, 18, 40).transpose(1,2)
  pred2 = torch.randn(256, 18, 40).transpose(1,2)

  l1 = emd_loss_ring(data1,pred1, r=2)   #!20211230
  print(l1)   #!20211230
  l2 = emd_loss_ring_3d(data2,pred2, r=2)   #!20211230
  print(l2)   #!20211230
# %%
