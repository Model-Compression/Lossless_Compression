from re import A


weight_num_list = [500,  500,  500]
    #                     [1000,  1000,  1000],  
    #                     [1500,  1500,  1500],
    #                     [2000,  2000,  2000],
    #                     [2500,  2500,  2500]]


a = (weight_num_list[1]*weight_num_list[2]+weight_num_list[0]*weight_num_list[1]+weight_num_list[2]*10)*32 + sum(weight_num_list)*32
print(a)


me_new = (input_num*weight_num_list[0]+weight_num_list[0]*weight_num_list[1]+weight_num_list[1]*weight_num_list[2]+10*weight_num_list[2])*(1-kesi)+(sum(weight_num_list))
    # print('MEM_new = ',me_new)

# weight_num_list = [2000, 2000, 1000]  # number for neurons for each layer
# weight_num_list = [1500,  1500,  1500]
# weight_num_list = [5000,  5000,  2000]
# weight_num_list = [5000,  5000,  5000]    
# weight_num_list = [10000,  10000,  5000]
# weight_num_list = [20000,  20000,  11000]
# weight_num_list = [500,  500,  500]
#                     [1000,  1000,  1000],  
#                     [1500,  1500,  1500],
#                     [2000,  2000,  2000],
#                     [2500,  2500,  2500]]
# weight_num_list = [1000,  1000, 1000]
# weight_num_list = [1500,  1500,  1500]
# weight_num_list = [2000,  2000,  2000]
# weight_num_list = [2500,  2500,  2500]
