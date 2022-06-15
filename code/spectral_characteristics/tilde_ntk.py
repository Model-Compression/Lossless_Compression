# from tilde_CK import calculate_CK_loop, calculate_CK_tilde, my_model, my_dataset_custome, my_activation_torch
# import numpy as np
# import torch.nn as nn
# import torch
# from torch.utils.data import DataLoader, Dataset
# import torch.optim as optim
# # from torch.
# from plot_eigen import plot_eigen
# # NTK loop here??????????????????
# def calculate_NTK_loop(model, data, loop):
#     [n,p] = data.shape
#     Phi = np.zeros((n, n))
#     model.cuda()
#     data.cuda()

#     for i in range(loop):
#         # initialize weight
#         for fc in model.fc_layers:
#             nn.init.normal_(fc.weight)
#         with torch.no_grad():
#             out = model(data).detach().cpu().numpy()
#             loss = nn.MSELoss(out, )
#         Phi_loop = out@out.T/loop
#         Phi = Phi + Phi_loop
#         # print (Phi)
    
#     return Phi

# def calculate_NTK_tilde(model, data, tau_zero):


# if __name__ == "__main__":
#     layer_num = 3                                                                 # layer number for network
#     input_num = 784                                                                    # input dimension for network 784/256
#     weight_num_list = [32, 32, 32]                                    # number for neurons for each layer
#     activation_list = [{'name' : 'ReLU', 'args' : None},
#                         {'name' : 'Binary_Zero', 'args' : {'s1':1, 's2': 2, 'b1': 1}},
#                         # {'name' : 't', 'args' : None}, 
#                         {'name' : 'ReLU', 'args' : None}] 
#                         # {'name' : 'ReLU', 'args' : None}]                               # activation for each layer, if with param, write as Binary_Zero here
#     loop = 100                                                                   # loop for estimation CK

#     #  define model 
#     model = my_model(layer_num = layer_num, input_num = input_num, weight_num_list = weight_num_list, activation_list = activation_list)
#     # model.cuda()

#     # load data
#     dataset = my_dataset_custome('MNIST', T=5000, cs=[0.5,0.5], selected_target=[6,8])
#     # dataset = my_dataset_custome('iid',T=8000, p=300, cs=[0.4,0.6],selected_target=[1,2])  #selected_target here used for calculate class number ,you can change to any other index with the same len(selected_target)

#     dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)    # shuffle????????????
#     data, _ = next(iter(dataloader))

#     # data.cuda()

#     T = dataset.T
#     P = np.eye(T) - np.ones((T,T))/T



#     # calculate NTK
#     NTK = calculate_NTK_loop(model, data)


#     # calculate NTK_tilde
#     NTK_tilde = calculate_NTK_tilde(model, data)

#     plot_eigen(NTK, NTK_tilde)

