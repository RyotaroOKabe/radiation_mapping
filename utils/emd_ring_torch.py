import torch
import numpy as np

# def tril_indices(n, k=0):
#     """Return the indices for the lower-triangle of an (n, m) array.
#     Works similarly to `np.tril_indices`
#     Args:
#       n: the row dimension of the arrays for which the returned indices will
#         be valid.
#       k: optional diagonal offset (see `np.tril` for details).
#     Returns:
#       inds: The indices for the triangle. The returned tuple contains two arrays,
#         each with the indices along one dimension of the array.
#     """
#     m1 = tf.tile(tf.expand_dims(tf.range(n), axis=0), [n, 1])
#     m2 = tf.tile(tf.expand_dims(tf.range(n), axis=1), [1, n])
#     mask = (m1 - m2) >= -k
#     ix1 = tf.boolean_mask(m2, tf.transpose(mask))
#     ix2 = tf.boolean_mask(m1, tf.transpose(mask))
#     return ix1, ix2

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
      # print('p: ', (torch.sum(p,dim=1,keepdim=True)))
      # print('p_hat: ', (torch.sum(p_hat,dim=1,keepdim=True)))
      # for k in range(p.shape[0]):
      #   suml = torch.sum(p[k,:])
      #   if suml <1:
      #     print('SUm error occurs with: ', p[k,:])
    #p=p/torch.sum(p,dim=1,keepdims=True) #!20211229
    #p_hat=p_hat/torch.sum(p_hat,dim=1,keepdims=True)
    # print("======test0=====") #!20220330
    # print(p_hat)  #!20220303 >> check!
    # p_hat_m=torch.mean(p_hat) #,dim=0) #!20220330 testing
    # print(p_hat_m) #!20220330 testing >> This part was no problem
    # p=p/torch.sum(p,dim=1,keepdim=True) #!20211229
    p=p/(torch.sum(p,dim=1,keepdim=True)+small) #!20221127
    # print(p)
    # p_m=torch.mean(p) #,dim=0) #!20220330 testing
    # print(p_m) #!20220330 testing >> Include nan
    # p_hat=p_hat/torch.sum(p_hat,dim=1,keepdim=True) #!20211229
    p_hat=p_hat/(torch.sum(p_hat,dim=1,keepdim=True)+small) #!20221127
    # print(p_hat)  #!20220303 >> check!
    # p_hat_m=torch.mean(p_hat) #,dim=0) #!20220330 testing
    # print(p_hat_m) #!20220330 testing
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
    # print("======test1=====") #!20220330
    # print(ecdf_p)  #!20220303 >> error!
    # ecdf_p_m=torch.mean(ecdf_p,dim=0) #!20220330 testing
    # print(ecdf_p_m) #!20220330 testing >> Include nan
    # print("======test2=====") #!20220330
    ecdf_p_hat=torch.stack(ecdf_p_hats)
    # print(ecdf_p_hat)  #!20220303  >> check!!
    # ecdf_p_hat_m=torch.mean(ecdf_p_hat,dim=0) #!20220330 testing
    # print(ecdf_p_hat_m) #!20220330 testing >> Include nan
    # #print(ecdf_p.get_shape())
    # print("======test3=====") #!20220330
    emd = torch.sum(torch.pow(torch.abs(ecdf_p - ecdf_p_hat), r), dim=2)
    # print(emd)  #!20220303
    # emd_m=torch.mean(emd,dim=0) #!20220330 testing
    # print(emd_m) #!20220330 testing  >> Include nan
    # print("======test4=====") #!20220330
    emd = torch.min(emd,dim=0)[0]
    # print(emd)
    # emd_m=torch.mean(emd,dim=0) #!20220330 testing
    # print(emd_m) #!20220330 testing >> Include nan
    # print("======test5=====") #!20220330
    #print(emd)  #!20220303 testing
    emd = torch.pow(emd, 1. / r)
    #print("====emd start===")
    # print(emd)
    # print(emd.shape)
    # print(type(emd))
    # emd_m=torch.mean(emd,dim=0) #!20220330 testing
    # print(emd_m) #!20220330 testing >> Include nan
    # #print(emd)  #!20220303 testing
    # print("======test6=====") #!20220330
    #print("====emd end===")
    emd=torch.mean(emd,dim=0)
    # print(emd)  #!20220303
    # emd_m=torch.mean(emd,dim=0) #!20220330 testing
    # print(emd_m) #!20220330 testing >> Include nan
    return emd

#%%
if __name__ == '__main__':
    from pyemd import emd
    data1=np.array([[0,0,1,0,0,0.1,0,0,0],[0,1,0,0,1,1,0,1,0]],dtype=np.float64).reshape(2,-1)
    data2=np.array([[0,1,0,1,0,0,0,1,0],[0,1,0,0,0,1,0,0,0]],dtype=np.float64).reshape(2,-1)

    # data1=np.array([[0,0,0,0,1,0,0,1,0,0,0]],dtype=np.float64).reshape(1,-1)
    # data2=np.array([[0,1,0,0,1,0,2,0,1,0,0]],dtype=np.float64).reshape(1,-1)
    a=data1[0].copy().reshape(1,-1)
    b=data2[0].copy().reshape(1,-1)

    #print data1
    #data1=data1/np.sum(data1,axis=1,keepdims=True)
    #data2=data2/np.sum(data2,axis=1,keepdims=True)
    #print data1

    #print(emd_loss_ring(torch.as_tensor(data1),torch.as_tensor(data2), r=2))  #!20211230
    print(emd_loss_ring(torch.from_numpy(data1),torch.from_numpy(data2), r=2))   #!20211230

    n=a.shape[1]
    M1=np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            #M1[i,j]=abs(i-j)
            M1[i,j]=min(abs(i-j),j+n-i,i+n-j)#**2

    # a=a[0]/a.sum()
    # b=b[0]/b.sum()
    # print(emd(a,b,M1))
# %%
