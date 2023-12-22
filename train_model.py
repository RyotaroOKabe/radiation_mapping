#%%
import torch
# import tensorboardX as tbx
# writer = tbx.SummaryWriter('runs')
from utils.dataset import get_output, FilterData2, load_data
from utils.unet import *
from utils.model import MyNet2, Model
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

# %%
#=========================set values here==========================
num_sources = 1
seg_angles = 64
epochs = 200
data_name = 'sq2_1_data'  #'230118-005847' # training data folder name
filter_name = 'sq2_1_filter'    #'230120-214857'   # filter folder name
save_name = 'sq2_1'  #f"{data_name}_{filter_name}"  # the name for saving the model
save_header = f"{save_dir}/{save_name}" 
path = f'./save/openmc_data/{data_name}'
filterpath = f'./save/openmc_filter/{filter_name}' 

filter_data2 = FilterData2(filterpath)
test_ratio = 0.1 # The number of data used for testing the model
k_fold = 5  # k-fold cross validation
print(save_name)
output_fun = get_output
net = MyNet2(seg_angles=seg_angles, filterdata=filter_data2).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)

kld_loss = torch.nn.KLDivLoss(size_average=None, reduction='batchmean')
loss_train = lambda  y, y_pred: emd_loss_ring(y, y_pred, r=2)
source_num, prob = [1 for _ in range(num_sources)], [1. for _ in range(num_sources)]

loss_val = lambda y, y_pred: emd_loss_ring(y, y_pred, r=1).item()
model = Model(net, loss_train, loss_val,reg=0.001)
train_set,test_set=load_data(test_ratio=test_ratio,test_size_gen=None,seg_angles=seg_angles,
                                output_fun=output_fun,path=path,source_num=source_num,prob=prob,seed=None)

optim = torch.optim.Adam([
    {"params": net.unet.parameters()},
    {"params": net.l1.weight1, 'lr': 3e-5},
    {"params": net.l1.weight2, 'lr': 3e-5},
    {"params": net.l1.bias1, 'lr': 3e-5},
    {"params": net.l1.bias2, 'lr': 3e-5},
    {"params": net.l1.Wn2, 'lr': 3e-5},
    {"params": net.l1.Wn1, 'lr': 3e-5}
    ], lr=0.001)

#=================================================================

#%%
# training
model.train(optim,train_set,test_set, epochs,batch_size=256,
            split_fold=k_fold,acc_func=None, verbose=10, save_dir=save_dir, 
            save_file=save_name)

