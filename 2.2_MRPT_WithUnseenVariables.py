
import sys
sys.path.append('..')

import io
import csv
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

from tqdm import tqdm
from torch import nn 
from constant import DataSplit 
from network import Mutual_Representation_PreTrain_Net, Mutual_MultiEn_MultiDe_FineTune_Net, Mutual_SensorToFeature_InterInference_LoadFeature_STD

# Specify the GPUs to use
device_ids = [1]
device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

N_EPOCH = 2000000
n_field_info = 36
n_baseF = 50 
n_cond = 11 #Length of the condition vector in U
N_P_Selected = 500

field_names = ['T', 'P', 'Vx', 'Vy', 'O2', 'CO2', 'H2O', 'CO', 'H2', 'NH3', 'HCN', 'NO']
Old_field_names = ['T', 'P', 'Vx', 'Vy', 'O2', 'CO2', 'H2O', 'CO', 'H2']

#________________________________________________________________________________________
Pretrain_Case_Num = 300
field_idx = 0 # The field used for sparse reconstruction
N_selected = 25  # Points to be extracted for Y_select as "sensors"
LOAD_INDICE = True
Y_INDICE_NAME = 'Y_select_indices_300_0_25.pickle'
hidden_sizes = [500, 500]
layer_sizes = [n_field_info * len(Old_field_names)] + hidden_sizes + [n_field_info]  
Final_layer_sizes = [n_field_info] + hidden_sizes + [n_field_info] 
#________________________________________________________________________________________

mode = False    #   True: finetune based on the pre-trained model / False: Using a newly trained DeepONet
N_PreTrain = 9 # Number of field that adopted in the pre-training phase, here it's up to 'H2'
Unseen_field = 'NO' # Define the field that will not be trained in this stage / Set 'NONE' if don't want to exclude a field

#Transformer layer parameters
num_heads = 6
num_layers = 1
NET_TYPE = int(0) 
                # 0 = [Self_MultiEn_MultiDe_FineTune_Net]; 
NET_SETTINGS = f'Downstream task of recovering unseen {Unseen_field} field, mode = {mode}\tN_Case = 30\tConNet=[n_field_info, 50, 50, n_field_info]\n'

NET_NAME = f'MRPT_FineTune_NO_{mode}'

PreTrained_Net_Name = 'net_MRPT_Standard_200_2_state_dict'
Load_file_path = 'Output_Net/Pre-training/{}.pth'.format(PreTrained_Net_Name)

P2F_Net_Name = 'net_MRPT_SensorToFeature_FromF0_N25_LoadFeature_STD_state_dict'
Load_P2F_Net_path = 'Output_Net/{}.pth'.format(P2F_Net_Name)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def get_data_iter(U, Y, Yin, Gin, G_Target, G_PreTrain, N, batch_size = 180): # random sampling in each epoch
    num_examples = len(U)
    num_points = Y.shape[1]
    indices = list(range(num_examples))
    np.random.shuffle(indices)  
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) 
        j = j.to(device)

        selected_points = torch.randperm(num_points)[:N].to(device)
        yield  U.index_select(0, j), Y.index_select(0, j).index_select(1, selected_points), Yin.index_select(0, j), Gin.index_select(0, j), G_Target.index_select(0, j).index_select(1, selected_points), G_PreTrain.index_select(0, j).index_select(1, selected_points)

def custom_mse_loss(output, target):
    losses = []
    for k in range(output.shape[2]):  # Loop over fields
        field_loss = F.mse_loss(output[:, :, k], target[:, :, k], reduction='mean')
        losses.append(field_loss)

    total_loss = torch.stack(losses).sum()

    return total_loss, losses

def unstandardize_features(standardized_vectors, mean, std):
    return standardized_vectors * std + mean

if __name__ == '__main__':

    with open(f'Loss_csv/train_test_loss_{NET_NAME}.csv', 'wt') as fp: 
        pass

    # Load the pretrained net
    PreTrained_net = Mutual_Representation_PreTrain_Net(n_field_info, n_baseF, num_heads, num_layers, num_fields=len(Old_field_names)).to(device)
    state_dict = torch.load(Load_file_path)
    PreTrained_net.load_state_dict(state_dict)
    print(f'I have loaded the MRPT net from {Load_file_path}')

    # Load the P2F net
    P2F_net = Mutual_SensorToFeature_InterInference_LoadFeature_STD(layer_sizes, Final_layer_sizes, PreTrained_net).to(device)
    state_dict = torch.load(Load_P2F_Net_path)
    P2F_net.load_state_dict(state_dict)
    print(f'I have loaded the Points-to-Feature net from {Load_P2F_Net_path}')

    with open('data_split/data_split_Multi_NO.pic', 'rb') as fp: 
        data_split = pickle.load(fp)

        U_train = data_split.U_train.to(device)
        Y_train = data_split.Y_train.to(device)
        G_train = data_split.G_train.to(device)

        U_test = data_split.U_test.to(device)
        Y_test = data_split.Y_test.to(device)
        G_test = data_split.G_test.to(device)

        n_inputF = U_train.shape[-1]
        n_pointD = Y_train.shape[-1]

        # Modify G_train based on the excluded field
        if Unseen_field in field_names:
            print('Preparing to exclude a field: ', Unseen_field)
            Unseen_field_index = field_names.index(Unseen_field)

            G_Target_train = G_train[:, :, Unseen_field_index:Unseen_field_index+1]
            G_Target_test = G_test[:, :, Unseen_field_index:Unseen_field_index+1]
            print('G_Target_train.shape is', G_Target_train.shape)
            print('G_Target_test.shape is', G_Target_test.shape)

            # Create G_Pretrain containing all fields except the excluded field
            G_Pretrain_train = G_train[:, :, :N_PreTrain]
            G_Pretrain_test = G_test[:, :, :N_PreTrain]

        else:
            print('excluded_field is not in field_names !!!')
            exit(0)

        if LOAD_INDICE is True:
            with open(f'Y_Indice/{Y_INDICE_NAME}', 'rb') as f:
                Y_select_indices = pickle.load(f)
                print(f'Loaded the Y_select_indices from {Y_INDICE_NAME}')
        else:
            Y_select_indices = torch.randperm(Y_train.size(1))[:N_selected].numpy()

        Yin_train = Y_train[:, Y_select_indices, :].to(device)
        Yin_test = Y_test[:, Y_select_indices, :].to(device)
        print('Y_train.shape = ', Y_train.shape)
        print('Yin_train.shape = ', Yin_train.shape)

        # Extract the temperature values (field_idx = 0) from G_train and G_test or other SELECTED field
        Gin_train = G_train[:, Y_select_indices, field_idx].unsqueeze(-1).to(device)  # Select the first field (temperature)
        Gin_test = G_test[:, Y_select_indices, field_idx].unsqueeze(-1).to(device)  # Select the first field (temperature)
        print('Gin_Train.shape = ', Gin_train.shape)
        # exit()

    field_info_train_tensors = {}
    field_info_test_tensors = {}
    mean_train_tensors = {}
    mean_test_tensors = {}
    std_train_tensors = {}
    std_test_tensors = {}

    with open(f'data_split/data_split_MRPT_Features_{Pretrain_Case_Num}_{field_idx}_{N_selected}_STD.pic', 'rb') as fp: 
        data_split = pickle.load(fp)

        LR_T_train       = data_split.LR_T_train.to(device)
        field_info_train_tensors[f'field_info_T'] = LR_T_train
        MEAN_T_train                              = data_split.MEAN_T_train.to(device)
        mean_train_tensors[f'field_info_T']       = MEAN_T_train
        STD_T_train                               = data_split.STD_T_train.to(device)
        std_train_tensors[f'field_info_T']        = STD_T_train

        LR_P_train       = data_split.LR_P_train.to(device)
        field_info_train_tensors[f'field_info_P']   = LR_P_train
        MEAN_P_train                                = data_split.MEAN_P_train.to(device)
        mean_train_tensors[f'field_info_P']         = MEAN_P_train
        STD_P_train                                 = data_split.STD_P_train.to(device)
        std_train_tensors[f'field_info_P']          = STD_P_train

        LR_Vx_train      = data_split.LR_Vx_train.to(device)
        field_info_train_tensors[f'field_info_Vx']  = LR_Vx_train
        MEAN_Vx_train                               = data_split.MEAN_Vx_train.to(device)
        mean_train_tensors[f'field_info_Vx']        = MEAN_Vx_train
        STD_Vx_train                                = data_split.STD_Vx_train.to(device)
        std_train_tensors[f'field_info_Vx']         = STD_Vx_train

        LR_Vy_train      = data_split.LR_Vy_train.to(device)
        field_info_train_tensors[f'field_info_Vy']  = LR_Vy_train
        MEAN_Vy_train                               = data_split.MEAN_Vy_train.to(device)
        mean_train_tensors[f'field_info_Vy']        = MEAN_Vy_train
        STD_Vy_train                                = data_split.STD_Vy_train.to(device)
        std_train_tensors[f'field_info_Vy']         = STD_Vy_train

        LR_O2_train      = data_split.LR_O2_train.to(device)
        field_info_train_tensors[f'field_info_O2']  = LR_O2_train
        MEAN_O2_train                               = data_split.MEAN_O2_train.to(device)
        mean_train_tensors[f'field_info_O2']        = MEAN_O2_train
        STD_O2_train                                = data_split.STD_O2_train.to(device)
        std_train_tensors[f'field_info_O2']         = STD_O2_train

        LR_CO2_train     = data_split.LR_CO2_train.to(device)
        field_info_train_tensors[f'field_info_CO2']  = LR_CO2_train
        MEAN_CO2_train                               = data_split.MEAN_CO2_train.to(device)
        mean_train_tensors[f'field_info_CO2']        = MEAN_CO2_train
        STD_CO2_train                                = data_split.STD_CO2_train.to(device)
        std_train_tensors[f'field_info_CO2']         = STD_CO2_train

        LR_H2O_train     = data_split.LR_H2O_train.to(device)
        field_info_train_tensors[f'field_info_H2O']  = LR_H2O_train
        MEAN_H2O_train                               = data_split.MEAN_H2O_train.to(device)
        mean_train_tensors[f'field_info_H2O']        = MEAN_H2O_train
        STD_H2O_train                                = data_split.STD_H2O_train.to(device)
        std_train_tensors[f'field_info_H2O']         = STD_H2O_train

        LR_CO_train      = data_split.LR_CO_train.to(device)
        field_info_train_tensors[f'field_info_CO']  = LR_CO_train
        MEAN_CO_train                               = data_split.MEAN_CO_train.to(device)
        mean_train_tensors[f'field_info_CO']        = MEAN_CO_train
        STD_CO_train                                = data_split.STD_CO_train.to(device)
        std_train_tensors[f'field_info_CO']         = STD_CO_train

        LR_H2_train      = data_split.LR_H2_train.to(device)
        field_info_train_tensors[f'field_info_H2']  = LR_H2_train
        MEAN_H2_train                               = data_split.MEAN_H2_train.to(device)
        mean_train_tensors[f'field_info_H2']        = MEAN_H2_train
        STD_H2_train                                = data_split.STD_H2_train.to(device)
        std_train_tensors[f'field_info_H2']         = STD_H2_train

        LR_Unified_train = data_split.LR_Unified_train.to(device)
        MEAN_Unified_train                             = data_split.MEAN_Unified_train.to(device)
        mean_train_tensors[f'Unified']                  = MEAN_Unified_train
        STD_Unified_train                              = data_split.STD_Unified_train.to(device)
        std_train_tensors[f'Unified']                   = STD_Unified_train 

        LR_T_test    = data_split.LR_T_test.to(device)
        field_info_test_tensors[f'field_info_T'] = LR_T_test
        MEAN_T_test                              = data_split.MEAN_T_test.to(device)
        mean_test_tensors[f'field_info_T']       = MEAN_T_test
        STD_T_test                               = data_split.STD_T_test.to(device)
        std_test_tensors[f'field_info_T']        = STD_T_test

        LR_P_test    = data_split.LR_P_test.to(device)
        field_info_test_tensors[f'field_info_P']   = LR_P_test
        MEAN_P_test                                = data_split.MEAN_P_test.to(device)
        mean_test_tensors[f'field_info_P']         = MEAN_P_test
        STD_P_test                                 = data_split.STD_P_test.to(device)
        std_test_tensors[f'field_info_P']          = STD_P_test

        LR_Vx_test       = data_split.LR_Vx_test.to(device)
        field_info_test_tensors[f'field_info_Vx']  = LR_Vx_test
        MEAN_Vx_test                               = data_split.MEAN_Vx_test.to(device)
        mean_test_tensors[f'field_info_Vx']        = MEAN_Vx_test
        STD_Vx_test                                = data_split.STD_Vx_test.to(device)
        std_test_tensors[f'field_info_Vx']         = STD_Vx_test

        LR_Vy_test       = data_split.LR_Vy_test.to(device)
        field_info_test_tensors[f'field_info_Vy']  = LR_Vy_test
        MEAN_Vy_test                               = data_split.MEAN_Vy_test.to(device)
        mean_test_tensors[f'field_info_Vy']        = MEAN_Vy_test
        STD_Vy_test                                = data_split.STD_Vy_test.to(device)
        std_test_tensors[f'field_info_Vy']         = STD_Vy_test

        LR_O2_test       = data_split.LR_O2_test.to(device)
        field_info_test_tensors[f'field_info_O2']  = LR_O2_test
        MEAN_O2_test                               = data_split.MEAN_O2_test.to(device)
        mean_test_tensors[f'field_info_O2']        = MEAN_O2_test
        STD_O2_test                                = data_split.STD_O2_test.to(device)
        std_test_tensors[f'field_info_O2']         = STD_O2_test

        LR_CO2_test      = data_split.LR_CO2_test.to(device)
        field_info_test_tensors[f'field_info_CO2']  = LR_CO2_test
        MEAN_CO2_test                               = data_split.MEAN_CO2_test.to(device)
        mean_test_tensors[f'field_info_CO2']        = MEAN_CO2_test
        STD_CO2_test                                = data_split.STD_CO2_test.to(device)
        std_test_tensors[f'field_info_CO2']         = STD_CO2_test

        LR_H2O_test      = data_split.LR_H2O_test.to(device)
        field_info_test_tensors[f'field_info_H2O']  = LR_H2O_test
        MEAN_H2O_test                               = data_split.MEAN_H2O_test.to(device)
        mean_test_tensors[f'field_info_H2O']        = MEAN_H2O_test
        STD_H2O_test                                = data_split.STD_H2O_test.to(device)
        std_test_tensors[f'field_info_H2O']         = STD_H2O_test

        LR_CO_test       = data_split.LR_CO_test.to(device)
        field_info_test_tensors[f'field_info_CO']  = LR_CO_test
        MEAN_CO_test                               = data_split.MEAN_CO_test.to(device)
        mean_test_tensors[f'field_info_CO']        = MEAN_CO_test
        STD_CO_test                                = data_split.STD_CO_test.to(device)
        std_test_tensors[f'field_info_CO']         = STD_CO_test

        LR_H2_test       = data_split.LR_H2_test.to(device)
        field_info_test_tensors[f'field_info_H2']  = LR_H2_test
        MEAN_H2_test                               = data_split.MEAN_H2_test.to(device)
        mean_test_tensors[f'field_info_H2']        = MEAN_H2_test
        STD_H2_test                                = data_split.STD_H2_test.to(device)
        std_test_tensors[f'field_info_H2']         = STD_H2_test

        LR_Unified_test  = data_split.LR_Unified_test.to(device)
        MEAN_Unified_test                             = data_split.MEAN_Unified_test.to(device)
        mean_test_tensors[f'Unified']                  = MEAN_Unified_test        
        STD_Unified_test                              = data_split.STD_Unified_test.to(device)
        std_test_tensors[f'Unified']                   = STD_Unified_test

        print('LR_Unified_train.shape is ', LR_Unified_train.shape)
        print('MEAN_Unified_train.shape is ', MEAN_Unified_train.shape)
        print('STD_Unified_train.shape is ', STD_Unified_train.shape, '\n')

    if (NET_TYPE == 0):
        # net = Mutual_MultiEn_MultiDe_FineTune_Net(n_inputF, n_field_info, n_baseF, PreTrained_net).to(device)
        net = Mutual_MultiEn_MultiDe_FineTune_Net(N_selected, n_field_info, n_baseF, PreTrained_net).to(device)
    else:
        print('Net Type is not correctly defined!!')
        exit()

    # Fix the parameters in the petrained net
    for param in net.PreTrained_net.parameters():
        param.requires_grad = False
    
    # Wrap the model with DataParallel
    net = nn.DataParallel(net, device_ids=device_ids)
    optimizer = optim.Adam(net.parameters(), lr=0.0005, weight_decay=1.0E-4)  

    # Set up early stopping parameters
    patience = 100
    best_combined_loss = float('5.0') #Initial threshold to determine early stopping
    counter = 0
    train_loss_weight = 0.25
    test_loss_weight = 0.75

    for epoch in range(N_EPOCH):
        train_loss = 0
        test_loss = 0
        train_batch_count = 0
        test_batch_count = 0
        for U, Y, Yin, Gin, G_T, G_P in get_data_iter(U_train, Y_train, Yin_train, Gin_train, G_Target_train, G_Pretrain_train, N = N_P_Selected):

            # print('Yin.device is', Yin.device)
            # print('Gin.device is', Yin.device)
            # for key, tensor in mean_train_tensors.items():
            #     print(f"Device of tensor '{key}': {tensor.device}")            
            # for key, tensor in std_train_tensors.items():
            #     print(f"Device of tensor '{key}': {tensor.device}")   

            Predicted_Features, Unified_Feature_output_train = P2F_net(Yin, Gin, num_heads, mean_train_tensors, std_train_tensors)
            Unified_Feature_output_train = unstandardize_features(Unified_Feature_output_train, mean_test_tensors[f'Unified'], std_test_tensors[f'Unified'])
            # print('Unified_Feature_output is ', Unified_Feature_output)
            # exit()

            optimizer.zero_grad()
            # output = net(U, Y, G_P, num_heads, mode)
            output = net(Unified_Feature_output_train, Y, Gin, num_heads, mode)
            loss = F.mse_loss(output, G_T, reduction='mean')
            train_loss += loss
            
            loss.backward()
            optimizer.step()
            train_batch_count += 1
        train_loss /= train_batch_count

        with torch.no_grad():  # Make sure to use no_grad for evaluation in test phase
            # for U, Y, G_T, G_P in get_data_iter(U_test, Y_test, G_Target_test, G_Pretrain_test, N = 500):
            for U, Y, Yin, Gin, G_T, G_P in get_data_iter(U_test, Y_test, Yin_test, Gin_test, G_Target_test, G_Pretrain_test, N = N_P_Selected):

                Predicted_Features, Unified_Feature_output_test = P2F_net(Yin, Gin, num_heads, mean_test_tensors, std_test_tensors)
                Unified_Feature_output_test = unstandardize_features(Unified_Feature_output_test, mean_test_tensors[f'Unified'], std_test_tensors[f'Unified'])

                # output = net(U, Y, G_P, num_heads, mode)
                output = net(Unified_Feature_output_test, Y, Gin, num_heads, mode)
                loss = F.mse_loss(output, G_T, reduction='mean')
                # loss = mse_loss(output, G_T)
                test_loss += loss
                test_batch_count += 1
            test_loss /= test_batch_count
        
        combined_loss = train_loss_weight*train_loss + test_loss_weight*test_loss

        # Print and write to CSV 
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{N_EPOCH}, Train Loss: {train_loss.item()}, Test Loss: {test_loss.item()}')

            # Write to CSV file
            with open(f'Loss_csv/train_test_loss_{NET_NAME}.csv', 'at', newline='') as fp:
                writer = csv.writer(fp, delimiter='\t')
                if ((epoch + 1) // 10 == 1):
                    fp.write(NET_SETTINGS)
                    header = ['Epoch', 'Train_Loss', 'Test_Loss']
                    writer.writerow(header)
                row_data = [epoch + 1, train_loss.item(), test_loss.item()] 
                writer.writerow(row_data)

            # Determine whether should trigger early stopping every 200 epochs
            if ( combined_loss < best_combined_loss ):
                best_combined_loss = combined_loss
                print(f'Best combined loss so far is {best_combined_loss}, still improving')
                counter = 0
                
                model_save_path = f'Output_Net/net_{NET_NAME}_state_dict.pth'
                torch.save(net.module.state_dict(), model_save_path)
                print('Successfully saved the latest best net at {}'.format(model_save_path))

            else:
                counter += 1
                print(f'Best combined loss so far is {best_combined_loss}, NOT further improving')
                print(f'The counter for triggering early stopping is {counter}')
                print()
                if counter >= patience:
                    print("Early stopping triggered")
                    break

