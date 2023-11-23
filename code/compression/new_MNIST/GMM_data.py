import torch
import torch.nn as nn
import numpy as np
from model import FCnet
import os
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import scipy
import torch.optim as optim
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import copy,math
from new_weight_act import act0, act1, binary0, binary1
from torch.optim.optimizer import Optimizer


# act0 = np.vectorize(act0)
# act1 = np.vectorize(act1) 

class get_dataset(Dataset):
    def __init__(self, data, label) -> None:
        super().__init__()
        self.data = data
        self.label = label
    
    def __getitem__(self, idx):
        return self.data[:,idx], self.label[idx]
    
    def __len__(self):
        return self.data.shape[1]

def gen_stat(testcase, selected_target=[-1,1],T = 8000,p = 256,cs=[0.5,0.5],means=None,covs=None):
    means=[]
    covs=[]
    if testcase == 'iid':
        for i in range(len(selected_target)):
            means.append(np.zeros(p)/np.sqrt(p))
            covs.append(np.eye(p)/p)     
    elif testcase == 'means':
        for i in range(len(selected_target)):
            means.append( np.concatenate( (np.zeros(i),4*np.ones(1),np.zeros(p-i-1)) )/np.sqrt(p) )
            # means.append( np.concatenate((np.zeros(1),4*np.ones(2),np.zeros(p-2-1)))/np.sqrt(p))
            covs.append(np.eye(p)/p)
    elif testcase == 'var':
        for i in range(len(selected_target)):
            means.append(np.zeros(p))
            covs.append(np.eye(p)*(1+8*i/np.sqrt(p)))
    elif testcase == 'mixed':
        for i in range(len(selected_target)):
            means.append( np.concatenate( (np.zeros(i),4*np.ones(1),np.zeros(p-i-1)) ) )
            covs.append((1+4*i/np.sqrt(p))*scipy.linalg.toeplitz( [(.4*i)**x for x in range(p)] )) 
    
    return means, covs

def gen_data(testbase, T = 8000,p = 256,cs=[0.5,0.5],means=None,covs=None,selected_target=[0,1,2,3,4,5,6,7,8,9],train = True):
    if testbase == 'GMM':
        X = np.array([]).reshape(p,0)    
        omega = np.array([]).reshape(p,0)
        y = []
        
        h = np.array([]).reshape(p,0)
        K = len(cs)
        for i in range(K):
            tmp = np.random.multivariate_normal(2*(i-K/2+.5)*means[i],covs[i],size=np.int(T*cs[i])).T
            X = np.concatenate((X,tmp),axis=1)
            omega = np.concatenate((omega, tmp - np.outer(means[i], np.ones((1,np.int(T*cs[i])))) ), axis=1)
            # h = np.concatenate((h,omega),axis = 1)
            # y = np.concatenate( (y,2*(i-K/2+.5)*np.ones(np.int(T*cs[i]))) )
            y = np.concatenate( (y,2*(i-K/2+.5)*np.ones(np.int(T*cs[i]))) )
        # aa = np.mean(np.diag(omega.T @ omega),axis=1)
        phi = np.diag(omega.T @ omega)
        '''均值和方差已经归一了'''
        # X = X/np.sqrt(p)
        # Omega = Omega/np.sqrt(p)
    elif testbase == 'MNIST' and train==True:
        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'data')
        if not os.path.isdir(root):
            os.makedirs(root)
        # if mode == 'train':
        mnist = dset.MNIST(root=os.path.join(root, 'train'),
                            train=True,
                            download=True)
        # else:
        #     mnist = dset.MNIST(root=os.path.join(root, 'test'),
        #                        train=False,
        #                        download=True)
        data, labels = mnist.data.view(mnist.data.shape[0], -1), mnist.targets

       
        selected_target = selected_target
        p = 784
        K = len(selected_target)

        # get the whole set of selected number
        data_full = []
        data_full_matrix = np.array([]).reshape(p, 0)
        # ind = 0
        for i in selected_target:
            locate_target_train = np.where(labels == i)[0]
            data_full.append(data[locate_target_train].T)
            data_full_matrix = np.concatenate(
                (data_full_matrix, data[locate_target_train].T), axis=1)
            # ind += 1

        # recentering and normalization to satisfy Assumption 1 and
        T_full = data_full_matrix.shape[1]
        mean_selected_data = np.mean(data_full_matrix, axis=1).reshape(p, 1)
        norm2_selected_data = np.sum(
            (data_full_matrix -
             np.mean(data_full_matrix, axis=1).reshape(p, 1))**2,
            (0, 1)) / T_full
        for i in range(K):
            data_full[i] = data_full[i] - mean_selected_data
            data_full[i] = data_full[i] * np.sqrt(p) / np.sqrt(
                norm2_selected_data)

        # # get the statistics of MNIST data
        # means = []
        # covs = []
        # for i in range(K):
        #     data_tmp = data_full[i]
        #     T_tmp = data_tmp.shape[1]
        #     means.append(np.mean(data_tmp.numpy(), axis=1).reshape(p, 1))
        #     covs.append((data_tmp @ (data_tmp.T) / T_tmp -
        #                  means[i] @ (means[i].T)).reshape(p, p))

        # data for train

        X = np.array([]).reshape(p, 0)
        y = []
        ind = 0
        for i in range(K):
            data_tmp = data_full[i]
            X = np.concatenate((X, data_tmp[:, range(int(cs[ind] * T))]),
                               axis=1)
            y = np.concatenate((y, ind * np.ones(int(T * cs[ind]))))
            ind += 1
        X = X / np.sqrt(p)

    elif testbase == 'MNIST' and train==False:
        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'data')
        if not os.path.isdir(root):
            os.makedirs(root)
        # if mode == 'train':
        mnist = dset.MNIST(root=os.path.join(root, 'test'),
                            train=False,
                            download=True)
        # else:
        #     mnist = dset.MNIST(root=os.path.join(root, 'test'),
        #                        train=False,
        #                        download=True)
        data, labels = mnist.data.view(mnist.data.shape[0], -1), mnist.targets

       
        selected_target = selected_target
        p = 784
        K = len(selected_target)

        # get the whole set of selected number
        data_full = []
        data_full_matrix = np.array([]).reshape(p, 0)
        # ind = 0
        for i in selected_target:
            locate_target_train = np.where(labels == i)[0]
            data_full.append(data[locate_target_train].T)
            data_full_matrix = np.concatenate(
                (data_full_matrix, data[locate_target_train].T), axis=1)
            # ind += 1

        # recentering and normalization to satisfy Assumption 1 and
        T_full = data_full_matrix.shape[1]
        mean_selected_data = np.mean(data_full_matrix, axis=1).reshape(p, 1)
        norm2_selected_data = np.sum(
            (data_full_matrix -
             np.mean(data_full_matrix, axis=1).reshape(p, 1))**2,
            (0, 1)) / T_full
        for i in range(K):
            data_full[i] = data_full[i] - mean_selected_data
            data_full[i] = data_full[i] * np.sqrt(p) / np.sqrt(
                norm2_selected_data)

        X = np.array([]).reshape(p, 0)
        y = []
        ind = 0
        for i in range(K):
            data_tmp = data_full[i]
            X = np.concatenate((X, data_tmp[:, range(int(cs[ind] * T))]),
                               axis=1)
            y = np.concatenate((y, ind * np.ones(int(T * cs[ind]))))
            ind += 1
        X = X / np.sqrt(p)

    return np.float32(X), np.float32(y)

def gd(X,Y,W,L,activation,mean,param=None):
    '''w: all weights matrix'''
    n = X.shape[1]
    gd = (L-1)*[0]
    D = (L-1)*[0]
    e = (L-1)*[0]
    y = (L-1)*[0]
    a = L*[0]  
    gd_mean = (L-1)*[0]   
    for j in range(n):
        e_l = L-2
        x = X[:,j]
        y[0] = W[0].T@x
        a[0] = x
        if activation == 'sin':
            # for j in range(n):
            #     x = X[:,j]
            #     y[0] = W[0].T@x
            #     a[0] = x
                for i in range(1,L-1):
                    y[i] = W[i].T@torch.sin(y[i-1])/torch.sqrt(torch.tensor(W[i-1].shape[1]))#'''不确定'''
                    a[i] = torch.sin(W[i-1].T@a[i-1])
                e[e_l] = -(Y[j]-y[L-2])  #不确定
                for i in range(0,L-2):
                    D[i] = torch.diag(torch.cos(y[i]))   
                while(e_l>=1):
                    e[e_l-1] = (W[e_l].T@D[e_l-1]).T@e[e_l]
                    e_l -= 1
                for i in range(1,L):
                    gd[i-1] += torch.outer(e[i-1],(a[i-1]).T)

        elif activation == 'ReLU':
            for i in range(1,L-1):
                y[i] = W[i].T@(torch.relu(y[i-1])-mean[i-1])/torch.sqrt(torch.tensor(W[i-1].shape[1]))
                a[i] = torch.relu(W[i-1].T@a[i-1])-mean[i-1]
            e[e_l] = -2*(Y[j]-y[L-2])/n
            for i in range(0,L-2):
                P = len(y[i])
                for j in range(P):
                    if y[i][j] < 0:
                        y[i][j] = 0
                    else:
                        y[i][j] = 1
                D[i] = torch.diag(y[i])
            while(e_l>=1):
                e[e_l-1] = (W[e_l].T@D[e_l-1]).T@e[e_l]
                e_l -= 1
            for i in range(1,L):
                gd[i-1] += torch.outer(e[i-1],(a[i-1]).T)

        elif activation == 'quan':
            for i in range(1,L-2):
                act = binary0(param[i-1][0],param[i-1][1],param[i-1][2])                
                a_out = torch.tensor(act(y[i-1]))-mean[i-1]/torch.sqrt(torch.tensor(W[i-1].shape[1]))
                y[i] = W[i].to(torch.float32).T@(a_out.to(torch.float32))
                a[i] = torch.tensor(act(y[i-1])-mean[i-1])
            act = binary1(param[L-3][0],param[L-3][1],param[L-3][2],param[L-3][3],param[L-3][4],param[L-3][5])
            y[L-2] = W[L-2].T@(act(y[L-3])-mean[L-3])/torch.sqrt(torch.tensor(W[L-2].shape[1]))
            a[L-2] = torch.tensor(act(y[L-3])-mean[L-3])
            e[e_l] = (Y[j]-y[L-2])/n
            for i in range(0,L-3):
                P = len(y[i])
                s1,s2,b = param[i]
                for j in range(P):
                    # if abs(s2-s1) > abs(b):
                    #     p1 = y[i][j] < s1+abs(b)/2
                    #     p2 = s2-abs(b)/2 < y[i][j]
                    #     if p1 == True:
                    #         y[i][j] = np.sign(b)*-1
                    #     if p2 == True:
                    #         y[i][j] = np.sign(b)*1
                    #     else:
                    #         y[i][j] = 0 
                    # else:
                    p1 =  y[i][j] < s1+abs(s2-s1)/2
                    # p2 = s2-abs(s2-s1)/2 <= y[i][j]
                    if p1:
                        y[i][j] = np.sign(b)*-1
                    # if p2:
                    #     y[i][j] = np.sign(b)*1
                    else:
                        # y[i][j] = 0  
                        y[i][j] = np.sign(b)*1                
                D[i] = torch.diag(y[i])                
            P = len(y[L-3])
            s1,s2,s3,s4,b1,b2 = param[L-3]
            # if abs(s2-s1) > (abs(b1)+abs(b2))/2 and abs(s3-s2) > abs(b2):
            #     for j in range(P):
            #         p1 = y[L-3][j] < s1+abs(b1)/2
            #         p2 = s2-abs(b2)/2 < y[L-3][j] < s2+abs(b2)/2
            #         p3 = s3-abs(b2)/2 < y[L-3][j] < s3+abs(b2)/2
            #         p4 = s4-abs(b1)/2 < y[L-3][j] 
            #         if p1:
            #             y[L-3][j] = np.sign(b1)*-1
            #         if p2:
            #             y[L-3][j] = np.sign(b1)*1
            #         if p3:
            #             y[L-3][j] = np.sign(b2)*-1
            #         if p4:
            #             y[L-3][j] = np.sign(b2)*1
            #         else:
            #             y[L-3][j] = 0                      
            # elif abs(s2-s1) > (abs(b1)+abs(b2))/2 and abs(s3-s2) < abs(b2):
            #     for j in range(P):            
            #         p1 =  y[L-3][j] < s1+abs(b1)/2
            #         p2 = s2-abs(b2)/2 < y[L-3][j] < s2+abs(s3-s2)/2
            #         p3 = s3-abs(s3-s2)/2 < y[L-3][j] < s3+abs(b2)/2
            #         p4 = s4-abs(b1)/2 < y[L-3][j] 
            #         if p1:
            #             y[L-3][j] = np.sign(b1)*-1
            #         if p2:
            #             y[L-3][j] = np.sign(b1)*1
            #         if p3:
            #             y[L-3][j] = np.sign(b2)*-1
            #         if p4:
            #             y[L-3][j] = np.sign(b2)*1
            #         else:
            #             y[L-3][j] = 0 
            # elif abs(s2-s1) < (abs(b1)+abs(b2))/2 and abs(s3-s2) > abs(b2):
            #     for j in range(P):
            #         p1 = y[L-3][j] < s1+abs(s2-s1)/2
            #         p2 = s2-abs(s2-s1)/2 < y[L-3][j] < s2+abs(b2)/2
            #         p3 = s3-abs(b2)/2 < y[L-3][j] < s3+abs(s4-s3)/2
            #         p4 = s4-abs(s4-s3)/2 < y[L-3][j]
            #         if p1:
            #             y[L-3][j] = np.sign(b1)*-1
            #         if p2:
            #             y[L-3][j] = np.sign(b1)*1
            #         if p3:
            #             y[L-3][j] = np.sign(b2)*-1
            #         if p4:
            #             y[L-3][j] = np.sign(b2)*1
            #         else:
            #             y[L-3][j] = 0 
            # else:
            for j in range(P):
                p1 = y[L-3][j] < s1+abs(s2-s1)/2
                p2 = s2-abs(s2-s1)/2 <= y[L-3][j] < s2+abs(s3-s2)/2
                p3 = s3-abs(s3-s2)/2 <= y[L-3][j] < s3+abs(s4-s3)/2
                p4 = s4-abs(s4-s3)/2 <= y[L-3][j]
                if p1:
                    y[L-3][j] = np.sign(b1)*-1
                elif p2:
                    y[L-3][j] = np.sign(b1)*1
                elif p3:
                    y[L-3][j] = np.sign(b2)*-1
                elif p4:
                    y[L-3][j] = np.sign(b2)*1
                # else:
                #     y[L-3][j] = 0 
            D[L-3] = torch.diag(y[L-3])
            while(e_l>=1):
                e[e_l-1] = (W[e_l].to(torch.float32).T@D[e_l-1]).T@e[e_l].to(torch.float32)
                e_l -= 1
            for i in range(1,L):
                gd[i-1] += torch.outer(e[i-1],(a[i-1]).T)
    for i in range(0,L-1):
        gd_mean[i] = gd[i]/n            
    return gd_mean

class OPT:
    def __init__(self, size, lr, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, n = None):
        self.n = n
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = []
        self.v = []
        for i in range(n):
            self.m.append(torch.zeros([size[i+1],size[i]]))
            self.v.append(torch.zeros([size[i+1],size[i]]))
        self.t = 0

    def update_SGD(self, W, grad = None):
        for i in range(self.n):
            W[i] = W[i].T
            W[i].T -= self.lr * grad[i].T
            W[i] = W[i].T
            W[i] = W[i].float()
        return W

    def update_Adam(self, W, grad = None):
        self.t += 1
        lr = self.lr
        # lr = self.lr * (1 - self.beta_2 ** self.t) ** 0.5 / (1 - self.beta_1 ** self.t)
        for i in range(self.n):
            g_t = grad[i]		
            self.m[i] = self.beta_1*self.m[i] + (1-self.beta_1)*g_t
            self.v[i] = self.beta_2*self.v[i] + (1-self.beta_2)*(g_t*g_t)
            self.m_cap = self.m[i]/(1-(self.beta_1**self.t))
            self.v_cap = self.v[i]/(1-(self.beta_2**self.t))
            W[i] = W[i].T			
            W[i] -= (lr*self.m_cap)/(torch.sqrt(self.v_cap)+self.epsilon)
            W[i] = W[i].T
            W[i] = W[i].float()
        return W

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch = 512
    batch_size = 512
    lr = 0.0002

    '''data_prepare'''
    selected_target=[-1,1]
    n = 512
    p = 256
    cs=[0.5,0.5]
    means, covs = gen_stat('means', selected_target=selected_target,T = n,p = p,cs=cs)
    X, Y, Omega, phi = gen_data(T=n,p=p,cs=cs,means=means,covs=covs)

    # print(np.mean(data[0,0:3999]),np.mean(data[1,4000:7999]),np.mean(data[3,:]))
    dataset = get_dataset(data = X,label = Y)
    train_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    
