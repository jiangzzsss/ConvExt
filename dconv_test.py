import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from ms_utils import tracktime
from dconv import DConv2d
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
device = torch.device('cuda:0')
n_convs = 1
BIAS = False
class Net(nn.Module):
    def __init__(self, conv) -> None:
        super(Net, self).__init__()
        layers = []
        for i in range(1,1+n_convs):
            # if(i%2==0):
            #     BIAS=True
            layers.append(nn.Sequential(
                conv(3, i*3, 3, stride=2, padding=1, bias=BIAS),
                nn.ReLU(inplace=True)
                ))
        self.layers = nn.Sequential(*layers)
    def forward(self, x_list):
        outs = 0
        for i, layer in enumerate(self.layers):
            outs += torch.sum(layer(x_list[i]))
        return outs

class DNet(nn.Module):
    def __init__(self, conv) -> None:
        super(DNet, self).__init__()
        layers = []
        for i in range(1,1+n_convs):
            # if(i%2==0):
            #     BIAS=True
            layers.append(nn.Sequential(
                conv(3, i*3, 3, stride=2, padding=1, bias=BIAS),
                nn.ReLU(inplace=False)
                ))
        self.layers = nn.Sequential(*layers)
    def forward(self, x_list):
        outs = 0
        for i, layer in enumerate(self.layers):
            outs += torch.sum(layer(x_list[i]))
        return outs

def test_dconv(iters = 5,conv=None):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    dnet = DNet(conv=conv).to(device)
    optimizer = optim.SGD(dnet.parameters(), lr = 0.01, momentum=0.9)
    time_lis = []
    res_lis = []
    grad_lis = []
    x_sz = 10
    x_batch = [[torch.randn((i+2,3,32,32), device=device, requires_grad=True) for _ in range(n_convs)] for i in range(x_sz)]
    for i in range(iters):
        optimizer.zero_grad()
        tracktime.cuda_record_start('dconv')
        y = dnet(x_batch[i%x_sz])
        loss = y.sum()
        res_lis.append(loss)
        loss.backward()
        grad_lis.append(x_batch[0][0].grad)
        _t = tracktime.cuda_record_end('dconv')
        time_lis.append(_t)
        optimizer.step()
    return sum(time_lis), x_batch[0], res_lis, grad_lis

def test_conv(iters = 5, conv=None):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    net = Net(conv=conv).to(device)
    optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)
    time_lis = []
    res_lis = []
    grad_lis = []
    x_sz = 10
    x_batch = [[torch.randn((i+2,3,32,32), device=device, requires_grad=True) for _ in range(n_convs)] for i in range(x_sz)]
    for i in range(iters):
        optimizer.zero_grad()
        tracktime.cuda_record_start('conv')
        y = net(x_batch[i%x_sz])
        loss = y.sum()
        res_lis.append(loss)
        loss.backward()
        grad_lis.append(x_batch[0][0].grad)
        _t = tracktime.cuda_record_end('conv')
        time_lis.append(_t)
        optimizer.step()
    return sum(time_lis), x_batch[0], res_lis, grad_lis

test_iters = 300
_t, x_batch, res_lis, grad_lis = test_conv(iters=test_iters, conv=nn.Conv2d)
_td, xd_batch, res_lisd, grad_lisd = test_dconv(iters=test_iters, conv=DConv2d)

grad_diff_sum = 0
loss_diff_sum = 0
for y, yd in zip(res_lis, res_lisd):
    l_diff = torch.max(torch.abs(y - yd))
    loss_diff_sum += l_diff
    # print('loss diff max:', l_diff)
for g, gd in zip(grad_lis, grad_lisd):
    g_diff = torch.max(torch.abs(g - gd))
    grad_diff_sum += g_diff
    # print('grad diff max: ', g_diff)
print(f'conv bias={BIAS}')
print(f'loss diff:{loss_diff_sum}   grad diff:{grad_diff_sum}')
# print(f'test {test_iters} iters: dconv= {_td}ms     conv= {_t}ms')
