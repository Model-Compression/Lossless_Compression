import torch
import numpy as np
import torch.nn as nn #包含网络的基类，比如卷积层、池化层、激活函数、全连接层等，损失函数的调用
from new_weight_act import new_act
import matplotlib.pyplot as plt

#定义一个类，这个类继承nn.module
class FCnet(nn.Module):
    def __init__(self, input, f1, f2, f3, f4):
        super(FCnet,self).__init__()
        self.F1 = nn.Linear(input,f1)
        self.R1 = nn.ReLU()
        self.F2 = nn.Linear(f1,f2)
        self.R2 = nn.ReLU()
        self.F3 = nn.Linear(f2,f3)
        self.R3 = nn.ReLU()
        self.OUT = nn.Linear(f3,f4)
        self.S = nn.Softmax(dim=1)
    def forward(self,x):
        # x = x.T
        x = self.F1(x)
        x = self.R1(x)
        x = x / torch.sqrt(torch.tensor(self.F1.weight.size(0)))
        x = self.F2(x)
        x = self.R2(x)
        x = x / torch.sqrt(torch.tensor(self.F2.weight.size(0)))
        x = self.F3(x)
        x = self.R3(x)
        x = x / torch.sqrt(torch.tensor(self.F3.weight.size(0)))
        x = self.OUT(x)
        x = self.S(x)
        return x
    def initialize(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight.data)
                # m.requires_grad_(True)
    def spar_init(self,kesi=0.5):
        spar_W = []
        for m in self.modules():
            if isinstance(m,nn.Linear): 
                init = np.zeros(len(m.weight.data.flatten()))
                init[:round(1 / 2 * (1 - kesi) * init.size)] = 1 / np.sqrt(1 - kesi)
                init[round(1 / 2 * (1 - kesi) * init.size):2 *round(1 / 2 * (1 - kesi) * init.size)] = -1 / np.sqrt(1 - kesi)
                np.random.shuffle(init)
                init = torch.tensor(init.reshape(m.weight.data.shape)).float() 
                m.weight = torch.nn.Parameter(init, requires_grad=True)
                spar_W.append(torch.abs(init * np.sqrt(1 - kesi)))
        return spar_W

class model_init(nn.Module):
    def __init__(self, input, f1, f2, f3, f4, arg, bias = False):    
        super(model_init,self).__init__()
        self.F1 = nn.Linear(input,f1)
        self.R1 = learnable_activation(arg[0])
        self.F2 = nn.Linear(f1,f2)
        self.R2 = learnable_activation(arg[1])
        self.F3 = nn.Linear(f2,f3)
        self.R3 = learnable_activation(arg[2])
        self.OUT = nn.Linear(f3,f4)
        self.S = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.F1(x)
        x = self.R1(x)
        x = x / torch.sqrt(torch.tensor(self.F1.weight.size(0)))
        x = self.F2(x)
        x = self.R2(x)
        x = x / torch.sqrt(torch.tensor(self.F2.weight.size(0)))
        x = self.F3(x)
        x = self.R3(x)
        x = x / torch.sqrt(torch.tensor(self.F3.weight.size(0)))

        # plt.clf()
        # plt.hist(x.detach().numpy().reshape(1,-1)[0],bins=100)
        # plt.savefig("plot/last_x")
        # plt.show()


        x = self.OUT(x)
        # x = x / torch.sqrt(torch.tensor(self.OUT.weight.size(0)))
        x = self.S(x)
        return x
    def initialize(self,kesi):
        for m in self.modules():
            if isinstance(m,nn.Linear): 
                init = np.zeros(len(m.weight.data.flatten()))
                init[:round(1 / 2 * (1 - kesi) * init.size)] = 1 / np.sqrt(1 - kesi)
                init[round(1 / 2 * (1 - kesi) * init.size):2 *round(1 / 2 * (1 - kesi) * init.size)] = -1 / np.sqrt(1 - kesi)
                np.random.shuffle(init)
                init = torch.tensor(init.reshape(m.weight.data.shape)).float() 
                m.weight = torch.nn.Parameter(init, requires_grad=True)
    def spar_init(self,kesi=0.5):
        spar_W = []
        for m in self.modules():
            if isinstance(m,nn.Linear): 
                init = np.zeros(len(m.weight.data.flatten()))
                init[:round(1 / 2 * (1 - kesi) * init.size)] = 1 / np.sqrt(1 - kesi)
                init[round(1 / 2 * (1 - kesi) * init.size):2 *round(1 / 2 * (1 - kesi) * init.size)] = -1 / np.sqrt(1 - kesi)
                np.random.shuffle(init)
                init = torch.tensor(init.reshape(m.weight.data.shape)).float() 
                m.weight = torch.nn.Parameter(init, requires_grad=True)
                spar_W.append(torch.abs(init * np.sqrt(1 - kesi)))
        return spar_W  
        
class learnable_activation(nn.Module):
    def __init__(self, *arg):
        super(learnable_activation, self).__init__()
        self.l = len(arg[0])
        if len(arg[0])==4:
            self.s1 = torch.nn.Parameter(torch.tensor(arg[0][0]), requires_grad=False)
            self.s2 = torch.nn.Parameter(torch.tensor(arg[0][1]), requires_grad=False)
            self.amp1 = torch.nn.Parameter(torch.tensor(arg[0][2]-arg[0][3]), requires_grad=True)
            self.amp2 = torch.nn.Parameter(torch.tensor(arg[0][2]-arg[0][3]), requires_grad=True)
            self.amp3 = torch.nn.Parameter(torch.tensor(-arg[0][3]), requires_grad=True)
        else:
            self.s1 = torch.nn.Parameter(torch.tensor(arg[0][0]), requires_grad=False)
            self.s2 = torch.nn.Parameter(torch.tensor(arg[0][1]), requires_grad=False)
            self.s3 = torch.nn.Parameter(torch.tensor(arg[0][2]), requires_grad=False)
            self.s4 = torch.nn.Parameter(torch.tensor(arg[0][3]), requires_grad=False)
            self.amp1 = torch.nn.Parameter(torch.tensor(arg[0][4]-arg[0][6]), requires_grad=True)
            self.amp2 = torch.nn.Parameter(torch.tensor(arg[0][5]-arg[0][6]), requires_grad=True)
            self.amp3 = torch.nn.Parameter(torch.tensor(arg[0][4]-arg[0][6]), requires_grad=True)
            self.amp4 = torch.nn.Parameter(torch.tensor(-arg[0][6]), requires_grad=True)
            self.amp5 = torch.nn.Parameter(torch.tensor(-arg[0][6]), requires_grad=True)
    def forward(self, input):
        if self.l == 4:
            return new_act.apply(input, self.s1, self.s2, self.amp1, self.amp2, self.amp3)
        else:
            return new_act.apply(input, self.s1, self.s2, self.s3, self.s4, self.amp1, self.amp2, self.amp3, self.amp4, self.amp5)

def ternary_weights(initialization_way, kesi, W):   
    init = np.zeros(len(W.flatten()))
    if initialization_way == 'ternary':
        init[:round(1 / 2 * (1 - kesi) * init.size)] = 1 / np.sqrt(1 - kesi)
        init[round(1 / 2 * (1 - kesi) * init.size):2 *round(1 / 2 * (1 - kesi) * init.size)] = -1 / np.sqrt(1 - kesi)
        # c = Counter(init)
        np.random.shuffle(init)
    ter_W = torch.tensor(init.reshape(W.shape)).float()
    return ter_W

def act0(x, s1, s2, a):
    if (x < s1 or x > s2):
        y = a
    else:
        y = 0
    return y

def act1(x, r1, r2, r3, r4, b1, b2):
    if (x<r1 or x>r4):
        y = b1
    elif (x>=r2 and x<=r3):
        y = b2
    else:
        y = 0
    return y 

def sigma_torch(x, a):
    '''
    define sigma function
    '''
    a = a.to(torch.float32)
    return torch.heaviside(x - a, torch.tensor(0.0))

def binary0(s1, s2, b1, b2, b3):
    return lambda x: b1 * sigma_torch(-x, -s1) + b2 * sigma_torch(x, s2) + b3 * (((sigma_torch(-x, -s1) + sigma_torch(x, s2)) - 1))

def binary1(s1, s2, s3, s4, b1, b2, b3, b4, b5):
    return lambda x: b1 * sigma_torch(-x, -s1) + b3 * sigma_torch(x, s4) + b2 * (((sigma_torch(-x, -s3) + sigma_torch(x, s2)) - 1)) + b4 * (((sigma_torch(-x, -s1) + sigma_torch(x, s2)) - 1)) + b5 * (((sigma_torch(-x, -s3) + sigma_torch(x, s4)) - 1))

def pre_acc(y):
    if y > 0:
        y = 1
    else:
        y = -1
    return y

class new_act(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, *arg):
        # ctx.save_for_backward(x, arg)
        l = len(arg)
        if l == 5:
            ctx.save_for_backward(x, arg[0], arg[1], arg[2], arg[3], arg[4])
            act = binary0(arg[0], arg[1], arg[2], arg[3], arg[4])
        else:
            ctx.save_for_backward(x, arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7], arg[8])
            act = binary1(arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7], arg[8])
        # aa = self.act
        return act(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        arg = ctx.saved_tensors
        # grad_x = 1*grad_output.clone()
        l = len(arg)
        if l == 6:
            grad_amp1 = grad_output.clone()*sigma_torch(-arg[0], -arg[1])
            grad_amp2 = grad_output.clone()*sigma_torch(arg[0], arg[2])
            grad_amp3 = grad_output.clone()*((sigma_torch(-arg[0], -arg[1]) + sigma_torch(arg[0], arg[2])) - 1)

            s1 = torch.sign(arg[5] - arg[3])
            s2 = torch.sign(arg[4] - arg[5])
            wx = gd_f(arg[0],arg[1],arg[2])
            grad_x = wx*grad_output.clone()
            return grad_x, None, None, None, None, None
        else:
            grad_amp1 = grad_output.clone()*sigma_torch(arg[0], arg[2])
            grad_amp2 = grad_output.clone()*((sigma_torch(-arg[0], -arg[3]) + sigma_torch(arg[0], arg[2])) - 1)
            grad_amp3 = grad_output.clone()*sigma_torch(arg[0], arg[4]) 
            grad_amp4 = grad_output.clone()*((sigma_torch(-arg[0], -arg[1]) + sigma_torch(arg[0], arg[2])) - 1)
            grad_amp5 = grad_output.clone()*((sigma_torch(-arg[0], -arg[3]) + sigma_torch(arg[0], arg[4])) - 1)
            wx = gd_l(arg[0],arg[1],arg[4])
            grad_x = wx*grad_output.clone()
            return grad_x, None, None, None, None, None, None, None, None, None


def gd_f(x,s1,s2,a1=None,a2=None):
    a = (s1+s2)/2
    # w = a * torch.ones(x.shape)
    # ww = torch.sign(x - w)
    # ww_d = a1 * (ww-1)/2
    # ww_g = a2 * (ww+1)/2
    # ww_out = ww_d + ww_g

    a1 = s1 - torch.abs(a)
    a2 = s2 + torch.abs(a)
    wl = 1 + torch.sign(a1 - x)
    wr = 1 + torch.sign(x - a2)
    ww = torch.abs(wl + wr - 2)/2

    return ww

def gd_l(x,s1,s2,a1=None,a2=None):
    a = (s1+s2)/2
    # w = a * torch.ones(x.shape)
    # ww = torch.sign(x - w)
    # ww_d = a1 * (ww-1)/2
    # ww_g = a2 * (ww+1)/2
    # ww_out = ww_d + ww_g

    a1 = s1 - torch.abs(a)
    a2 = s2 + torch.abs(a)
    wl = 1 + torch.sign(a1 - x)
    wr = 1 + torch.sign(x - a2)
    ww = torch.abs(wl + wr - 2)/2


    return ww
# test
if __name__ == "__main__":
    model = FCnet()
    a = torch.randn(1, 1, 28, 28)
    b = model(a)
    print(b)
