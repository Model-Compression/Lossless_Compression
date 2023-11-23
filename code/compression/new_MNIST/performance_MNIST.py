import torch
import torch.nn as nn
import numpy as np
from model import FCnet, model_init
from torch.utils.data import Dataset, DataLoader
from GMM_data import gen_data, gen_stat, get_dataset, gd, OPT
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matching import get_or_param, get_act0, get_act1, sovle_0, sovle_1


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 500
    batch_size = 256
    lr = 0.005
    L = 3

    '''data_prepare'''
    selected_target=[0,1,2,3,4,5,6,7,8,9]
    N_train = 50000
    N_test = 8000
    p = 784
    cs=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
   
    means, covs = gen_stat('means', selected_target=selected_target,T = N_train,p = p,cs=cs)
    X, Y = gen_data(testbase = 'MNIST', T=N_train, p=p, cs=cs, means=means, covs=covs, selected_target=selected_target, train=1)

    dataset = get_dataset(data = X,label = Y)
    train_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    X_test, Y_test = gen_data(testbase = 'MNIST', T=N_test, p=p, cs=cs, means=means, covs=covs, selected_target=selected_target, train=0)
    dataset_test = get_dataset(data = X_test,label = Y_test)
    test_dataset = DataLoader(dataset_test, batch_size=N_test, shuffle=True)

    '''data stastics and scales of original model'''
    tau_0 = np.sqrt(np.mean(np.diag(X.T @ X)))
  
    
    '''ReLU()'''
    or_pa = get_or_param(L, 'ReLU', tau_0)

    # '''matching to get light_model's activation by NTK_LC'''
    act_param = []
    tau_last = tau_0
    for i in range(0,L-1):
        new_param0 = get_act0(tau_last)
        a = sovle_0([or_pa[0][i], or_pa[1][i], or_pa[2][i]], new_param0, 1000)
        new_mean = new_param0[4](a[0],a[1],a[2])
        tau_last = np.sqrt(new_param0[3](a[0],a[1],a[2]))
        act_param.append(np.append(a, new_mean))

    b1 = np.sqrt((or_pa[2][L-1]**2 / or_pa[1][L-1]**2) + 4 * or_pa[3][L-1]**2)
    b2 = b1
    new_param1 = get_act1(b1,b2,tau_last)
    b = sovle_1([or_pa[0][L-1], or_pa[1][L-1], or_pa[2][L-1], or_pa[3][L]], new_param1, 1000)
    new_mean = new_param1[3](b[0],b[1],b[2],b[3])
    b.append(b1)
    b.append(b2)
    act_param.append(np.append(b, new_mean))
  
    
    # '''train'''
    loss_func = nn.CrossEntropyLoss()
    net1 = model_init(p, 512,512,1024, len(cs), act_param)
    # net1 = model_init(p, 1024,1024,2048, len(cs), act_param)
    # net1 = model_init(p, 2048,2048,4096, len(cs), act_param)
    # net1 = model_init(p, 256,256,512, len(cs), act_param)
    # net1 = model_init(p, 4096,4096,8192, len(cs), act_param)
    net1.to(device)
    net1.initialize(0)
    nn.init.normal_(net1.OUT.weight.data)
    optimizer1 = torch.optim.Adam(net1.OUT.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=200, gamma=0.1)
    net1.train()
    t = np.arange(0,epochs,1)
    loss_record = []
    acc = []
    acct = []
    for epoch in range(epochs):
        aa = 0
        for train_data, train_label in train_dataset:
            net1.train()
            net1.to(device)
            label_onehot = torch.zeros(train_label.shape[0], len(cs)).long().to(device)
            label_onehot.scatter_(dim=1,index=train_label.unsqueeze(dim=1).long().to(device),src=torch.ones(train_label.shape[0], len(cs)).long().to(device))
            optimizer1.zero_grad()
            pre = net1(train_data.to(device))
            ppre = (torch.argmax(pre,1)).to(torch.float32)
            loss = loss_func(pre, label_onehot.to(torch.float32).to(device))
            loss.requires_grad_(True)
            loss.backward()
            optimizer1.step()
            aa += torch.sum(ppre==train_label.to(device))
        
        # scheduler.step()
        loss_record.append(loss.item())
        aa = aa/N_train
        acc.append(aa)

        net1.eval()
        aat = 0
        for test_data, test_label in test_dataset:
            pre = net1(test_data.to(device))
            ppre = (torch.argmax(pre,1)).to(torch.float32)
            aat += torch.sum(ppre==test_label.to(device))
        aat = aat/N_test
        acct.append(aat)
        if epoch%20 == 1:
            print("epoch:",epoch)
            print("train:", aa)
            print("test:", aat) 
    print("new train done.")

