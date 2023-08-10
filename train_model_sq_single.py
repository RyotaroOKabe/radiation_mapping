#%%
"""
Created on 2022/12/27
@author: R.Okabe
"""
import torch
# import tensorboardX as tbx
# writer = tbx.SummaryWriter('runs')
from utils.dataset import get_output, FilterData1, FilterData2, load_data
from utils.unet import *
from utils.model import MyNet1, Model
from utils.emd_ring_torch import emd_loss_ring

GPU_INDEX = 0
USE_CPU = False
if torch.cuda.is_available() and not USE_CPU:
    DEFAULT_DEVICE = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    DEFAULT_DEVICE = torch.device("cpu")
DEFAULT_DTYPE = torch.double
save_dir = "./save/training"

#%%


# a_num = 2
# tetris_shape = 'L'
num_sources = 1
seg_angles = 64
epochs = 20
data_name = '230806-155301' #'230711-174017' # '221227-001319'
filter_name = '230807-194108'
#=========================================================
save_name = f"{data_name}_{filter_name}_{epochs}_{num_sources}_xxxx"   # save_name = f"{data_name}"
save_header = f"{save_dir}/{save_name}"
#save_name = f"openmc_tetris{tetris_shape}_{num_sources}src_{seg_angles}_ep{epochs}_bs256_20220821_v1.1"
#=========================================================
path = f'./save/openmc_data/{data_name}'
filterpath = f'./save/openmc_filter/{filter_name}'

filter_data1 = FilterData1(filterpath)
test_size = 400
k_fold = 5
print(save_name)
output_fun = get_output
net = MyNet1(seg_angles=seg_angles, filterdata=filter_data1)
net = net.to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)

kld_loss = torch.nn.KLDivLoss(size_average=None, reduction='batchmean')
loss_train = lambda  y, y_pred: emd_loss_ring(y, y_pred, r=2)
# loss_train = lambda y, y_pred: emd_loss_sinkhorn(y, y_pred, M2)
# loss_train = lambda y, y_pred: kld_loss(y_pred.log(),y
source_num, prob = [1 for _ in range(num_sources)], [1. for _ in range(num_sources)]

loss_val = lambda y, y_pred: emd_loss_ring(y, y_pred, r=1).item()
model = Model(net, loss_train, loss_val,reg=0.001)
train_set,test_set=load_data(test_size=test_size,train_size=None,test_size_gen=None,seg_angles=seg_angles,
                                output_fun=output_fun,path=path,source_num=source_num,prob=prob,seed=None)

optim = torch.optim.Adam([
    {"params": net.unet.parameters()},
    {"params": net.l1.weight1, 'lr': 3e-5},
    # {"params": net.l1.weight2, 'lr': 3e-5},
    {"params": net.l1.bias1, 'lr': 3e-5},
    # {"params": net.l1.bias2, 'lr': 3e-5},
    # {"params": net.l1.Wn2, 'lr': 3e-5},
    {"params": net.l1.Wn1, 'lr': 3e-5}
    ], lr=0.001)

# model.train(optim,train_set,test_set,epochs,batch_size=256,acc_func=None, verbose=10, save_name=save_name)
model.train(optim,train_set,test_set, epochs,batch_size=256,split_fold=k_fold,acc_func=None, verbose=10, save_name=save_header)
# model.save('save/models/' + save_name)

# model.plot_train_curve(save_name=save_header)
# model.plot_test(test_set,test_size,seg_angles=seg_angles,loss_fn=loss_val,save_dir=save_header)

# %%
