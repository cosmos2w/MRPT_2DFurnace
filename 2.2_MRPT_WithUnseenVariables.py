
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
from network import Mutual_Representation_PreTrain_Net, Mutual_MultiEn_MultiDe_FineTune_Net

# Specify the GPUs to use
device_ids = [0]
device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

N_EPOCH = 2000000
n_field_info = 36
n_baseF = 50 
n_cond = 11 #Length of the condition vector in U
N_P_Selected = 500

field_names = ['T', 'P', 'Vx', 'Vy', 'O2', 'CO2', 'H2O', 'CO', 'H2', 'NH3', 'HCN', 'NO']
Old_field_names = ['T', 'P', 'Vx', 'Vy', 'O2', 'CO2', 'H2O', 'CO', 'H2']

mode = True
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
Load_file_path = 'Output_Net/{}.pth'.format(PreTrained_Net_Name)

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

def get_data_iter(U, Y, G_Target, G_PreTrain, N, batch_size = 180): # random sampling in each epoch
    num_examples = len(U)
    num_points = Y.shape[1]
    indices = list(range(num_examples))
    np.random.shuffle(indices)  
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) 
        j = j.to(device)

        selected_points = torch.randperm(num_points)[:N].to(device)
        yield  U.index_select(0, j), Y.index_select(0, j).index_select(1, selected_points), G_Target.index_select(0, j).index_select(1, selected_points), G_PreTrain.index_select(0, j).index_select(1, selected_points)

def custom_mse_loss(output, target):
    losses = []
    for k in range(output.shape[2]):  # Loop over fields
        field_loss = F.mse_loss(output[:, :, k], target[:, :, k], reduction='mean')
        losses.append(field_loss)

    total_loss = torch.stack(losses).sum()

    return total_loss, losses

if __name__ == '__main__':

    with open(f'Loss_csv/train_test_loss_{NET_NAME}.csv', 'wt') as fp: 
        pass

    # Load the pretrained net
    PreTrained_net = Mutual_Representation_PreTrain_Net(n_field_info, n_baseF, num_heads, num_layers, num_fields=len(Old_field_names)).to(device)
    state_dict = torch.load(Load_file_path)
    PreTrained_net.load_state_dict(state_dict)

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

    num_fields = len(field_names)
    if (NET_TYPE == 0):
        net = Mutual_MultiEn_MultiDe_FineTune_Net(n_inputF, n_field_info, n_baseF, PreTrained_net).to(device)
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
        for U, Y, G_T, G_P in get_data_iter(U_train, Y_train, G_Target_train, G_Pretrain_train, N = N_P_Selected):

            optimizer.zero_grad()
            output = net(U, Y, G_P, num_heads, mode)
            loss = F.mse_loss(output, G_T, reduction='mean')
            train_loss += loss
            
            loss.backward()
            optimizer.step()
            train_batch_count += 1
        train_loss /= train_batch_count

        with torch.no_grad():  # Make sure to use no_grad for evaluation in test phase
            for U, Y, G_T, G_P in get_data_iter(U_test, Y_test, G_Target_test, G_Pretrain_test, N = 500):
                output = net(U, Y, G_P, num_heads, mode)
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

