
import io
import csv
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

from torch import nn 
from constant import DataSplit 
from network import CoFFe_PreTrain_Net_ParallelMode

# Specify the GPUs to use
device_ids = [0]
device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

#__________________________PARAMETERS_________________________________
N_EPOCH = 10000
BATCH_SIZE = 180
N_P_Selected = 200   # The number of random sparse sensors in each pre-training epoch
n_field_info = 36    # The dimension of latent features to be extracted from fied distribution data
n_baseF = 50         # The dimension of output layers of DeepONet-based decoder

Unified_Weight = 5.0 # Contribution of the unified feature

# FieldAttention layer parameters
num_heads = 6
num_layers = 1

field_names = ['T', 'P', 'Vx', 'Vy', 'O2', 'CO2', 'H2O', 'CO', 'H2']

# You can put important nots here, will be recorded in the output loss file
NET_SETTINGS = f'Unified_Weight = {Unified_Weight}\tn_field_info = {n_field_info}\tMultiHeadAttention={num_heads} & layer={num_layers}\tn_baseF = {n_baseF}\n'
NET_NAME     = f'CoFFe_ParallelMode'
#____________________________________________________________________

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def get_data_iter(U, Y, G, N, batch_size = BATCH_SIZE): # random sampling in each epoch
    num_examples = len(U)
    num_points = Y.shape[1]
    indices = list(range(num_examples))
    np.random.shuffle(indices)  
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) 
        j = j.to(device)

        selected_points = torch.randperm(num_points)[:N].to(device)
        yield  U.index_select(0, j), Y.index_select(0, j).index_select(1, selected_points), G.index_select(0, j).index_select(1, selected_points)

def field_mse_loss(output, target):
    losses = []
    for k in range(output.shape[2]):  # Loop over fields
        field_loss = F.mse_loss(output[:, :, k], target[:, :, k], reduction='mean')
        losses.append(field_loss)

    total_loss = torch.stack(losses).sum()

    return total_loss, losses

if __name__ == '__main__':

    with open(f'Loss_csv/train_test_loss_Parallel/train_test_loss_{NET_NAME}.csv', 'wt') as fp: 
        pass

    with open(f'Loss_csv/train_test_loss_Parallel/train_test_loss_{NET_NAME}_Unified.csv', 'wt') as fp: 
        pass

    with open('data_split/data_split.pic', 'rb') as fp: 
        data_split = pickle.load(fp)

        U_train = data_split.U_train.to(device)
        Y_train = data_split.Y_train.to(device)
        G_train = data_split.G_train.to(device)
        print("U_train.shape, Y_train.shape, G_train.shape", U_train.shape, Y_train.shape, G_train.shape)

        U_test = data_split.U_test.to(device)
        Y_test = data_split.Y_test.to(device)
        G_test = data_split.G_test.to(device)
        print("U_test.shape, Y_test.shape, G_test.shape", U_test.shape, Y_test.shape, G_test.shape)

        field_weights = torch.tensor([1.0] * len(field_names))
        field_weights = field_weights.to(device)

    num_fields = len(field_names) 
    net = CoFFe_PreTrain_Net_ParallelMode(n_field_info, n_baseF, num_heads, num_layers, num_fields).to(device)

    net = nn.DataParallel(net, device_ids=device_ids)
    net.apply(weights_init)

    optimizer = optim.Adam(net.parameters(), lr=0.0005)  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=1000, verbose=True, min_lr=0.00020)

    # Set up report and early stopping parameters
    N_REPORT = 20
    patience = 500
    best_combined_loss = float('5.0') # Initial threshold to determine early stopping
    counter = 0
    train_loss_weight = 0.15
    test_loss_weight = 0.85

    for epoch in range(N_EPOCH):

        Total_train_loss_Data = 0 # the total losses from all generated fields from each field: Loss_T = Sigma(i) Sigma(j) Field_j(Generated from Field_i)
        Total_test_loss_Data = 0
        Total_train_loss_Data_Unified = 0 # the total losses from all generated fields from each field: Loss_T = Sigma(i) Sigma(j) Field_j(Generated from Field_i)
        Total_test_loss_Data_Unified = 0
        train_loss = torch.zeros(len(field_names), device=device)  # [len(field_names), len(field_names)]
        test_loss = torch.zeros(len(field_names), device=device)  # [len(field_names), len(field_names)]

        train_loss_Unified = torch.zeros(len(field_names), device=device)  # [len(field_names), len(field_names)]
        test_loss_Unified = torch.zeros(len(field_names), device=device)  # [len(field_names), len(field_names)]

        train_batch_count = 0
        test_batch_count = 0

        for U, Y, G in get_data_iter(U_train, Y_train, G_train, N = N_P_Selected):
            
            optimizer.zero_grad()
            loss = 0
            output_list = net(U, Y, G, num_heads)
            output_stacked = torch.stack(output_list, dim=0)

            output = output_stacked[0, :, :, :]
            output_Unified = output_stacked[1, :, :, :]
            
            loss_data, losses = field_mse_loss(output, G)
            Total_train_loss_Data += loss_data
            loss += loss_data
            
            loss_data_Unified, losses_Unified = field_mse_loss(output_Unified, G)
            Total_train_loss_Data_Unified += loss_data_Unified
            loss += Unified_Weight * loss_data_Unified

            for j, loss_item in enumerate(losses):
                train_loss[j] += loss_item
            for j, loss_item in enumerate(losses_Unified):
                train_loss_Unified[j] += loss_item
            
            loss.backward()            
            optimizer.step()
            scheduler.step(loss)
            train_batch_count += 1

        train_loss /= train_batch_count
        Total_train_loss_Data /= train_batch_count
        train_loss_Unified /= train_batch_count
        Total_train_loss_Data_Unified /= train_batch_count

        # Print and write to CSV 
        if (epoch + 1) % N_REPORT == 0:

            with torch.no_grad():  # Make sure to use no_grad for evaluation in test phase
                for U, Y, G in get_data_iter(U_test, Y_test, G_test, N = 2000):
                    output_list = net(U, Y, G, num_heads)
                    output_stacked = torch.stack(output_list, dim=0)
                    output = output_stacked[0, :, :, :]
                    output_Unified = output_stacked[1, :, :, :]
                    
                    loss_data, losses = field_mse_loss(output, G)
                    Total_test_loss_Data += loss_data
                    loss_data_Unified, losses_Unified = field_mse_loss(output_Unified, G)
                    Total_test_loss_Data_Unified += loss_data_Unified

                    for j, loss_item in enumerate(losses):
                        test_loss[j] += loss_item
                    for j, loss_item in enumerate(losses_Unified):
                        test_loss_Unified[j] += loss_item
                    test_batch_count += 1
                
                test_loss /= test_batch_count
                Total_test_loss_Data /= test_batch_count
                test_loss_Unified /= test_batch_count
                Total_test_loss_Data_Unified /= test_batch_count
            
            combined_loss = train_loss_weight*Total_train_loss_Data_Unified + test_loss_weight*Total_test_loss_Data_Unified

            print(f'Epoch {epoch+1}/{N_EPOCH}, Total Train Loss: {Total_train_loss_Data.item()}, Total Test Loss: {Total_test_loss_Data.item()}')
            for field_idx, field_name in enumerate(field_names):
                print(f'Train Loss for {field_name}: {train_loss[field_idx].item()}, Test Loss for {field_name}: {test_loss[field_idx].item()}')
            print()
            print(f'Epoch {epoch+1}/{N_EPOCH}, Unified Total Train Loss: {Total_train_loss_Data_Unified.item()}, Unifed Total Test Loss: {Total_test_loss_Data_Unified.item()}')
            for field_idx, field_name in enumerate(field_names):
                print(f'Unified Train Loss for {field_name}: {train_loss_Unified[field_idx].item()}, Unified Test Loss for {field_name}: {test_loss_Unified[field_idx].item()}')

            # Write to CSV file
            with open(f'Loss_csv/train_test_loss_Parallel/train_test_loss_{NET_NAME}.csv', 'at', newline='') as fp:
                writer = csv.writer(fp, delimiter='\t')
                if ((epoch + 1) // N_REPORT == 1):
                    fp.write(NET_SETTINGS)
                    header = ['Epoch', 'Overall_Train_Loss', 'Overall_Test_Loss']
                    interleaved_field_names = [fn for field_name in field_names for fn in (f'Train_{field_name}_loss', f'Test_{field_name}_loss')]
                    header.extend(interleaved_field_names)
                    writer.writerow(header)
                row_data = [epoch + 1, Total_train_loss_Data.item(), Total_test_loss_Data.item()] 
                
                for train_loss, test_loss in zip(train_loss, test_loss):
                    row_data.append(f'{train_loss.item()}')
                    row_data.append(f'{test_loss.item()}')
                writer.writerow(row_data)
            
            with open(f'Loss_csv/train_test_loss_Parallel/train_test_loss_{NET_NAME}_Unified.csv', 'at', newline='') as fp:
                writer = csv.writer(fp, delimiter='\t')
                if ((epoch + 1) // N_REPORT == 1):
                    fp.write(NET_SETTINGS)
                    header = ['Epoch', 'Overall_Train_Loss', 'Overall_Test_Loss']
                    interleaved_field_names = [fn for field_name in field_names for fn in (f'Train_{field_name}_loss', f'Test_{field_name}_loss')]
                    header.extend(interleaved_field_names)
                    writer.writerow(header)
                row_data = [epoch + 1, Total_train_loss_Data_Unified.item(), Total_test_loss_Data_Unified.item()] 
                
                for train_loss, test_loss in zip(train_loss_Unified, test_loss_Unified):
                    row_data.append(f'{train_loss.item()}')
                    row_data.append(f'{test_loss.item()}')
                writer.writerow(row_data)

            if ( combined_loss < best_combined_loss ):
                best_combined_loss = combined_loss
                print(f'Best combined loss so far is {best_combined_loss}, still improving')
                counter = 0
                
                model_save_path = 'Output_Net/net_{}_state_dict.pth'.format(NET_NAME)
                torch.save(net.module.state_dict(), model_save_path)
                print('Successfully saved the latest best net at {}'.format(model_save_path))
            else:
                counter += 1
                print(f'Best combined loss so far is {best_combined_loss}, NOT further improving')
                print(f'The counter for triggering early stopping is {counter}\n')
                if counter >= patience:
                    print("Early stopping triggered")
                    break

