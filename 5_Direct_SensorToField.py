# In this version, the code will directly perform multi-field reconstruction

import sys
sys.path.append('..')

import io
import csv
import numpy as np 
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

from tqdm import tqdm
from torch import nn 
from constant import DataSplit 
from network import  Direct_SensorToField

# Specify the GPUs to use
device_ids = [2]
device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

N_EPOCH = 200000
Case_Num = 300
n_baseF = 40 
n_cond = 9 #Length of the condition vector in U
field_names = ['T', 'P', 'Vx', 'Vy', 'O2', 'CO2', 'H2O', 'CO', 'H2']
field_idx = 1 # The field used for sparse reconstruction

N_selected = 50  # Points to be extracted for Y_select as "sensors"
N_P_Selected = 2000 # Points to evaluate loss in each epoch

NET_TYPE = int(0) 
                # 0 = [Direct_SensorToField]; 
NET_SETTINGS = f'Direct training sensor points to field task via DeepONet\tn_baseF = {n_baseF}\tN_selected = {N_selected}\tPositionNet([2, 50, 50, 50, n_base])\tConditionNet([n_sensors, 50, 50, n_base])\n'
NET_NAME = [f'Direct_SensorToField_From{field_idx}_NP{N_selected}']

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

def get_data_iter(U, Y, G, Yin, Gin, batch_size = 360): # random sampling in each epoch
    num_examples = len(U)
    num_points = Y.shape[1]
    indices = list(range(num_examples))
    np.random.shuffle(indices)  
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        j = j.to(device)

        selected_points = torch.randperm(num_points)[:N_P_Selected].to(device)
        yield  U.index_select(0, j), Y.index_select(0, j).index_select(1, selected_points), G.index_select(0, j).index_select(1, selected_points), Yin.index_select(0, j), Gin.index_select(0, j)

def custom_mse_loss(output, target, field_weights):

    num_fields = target.size(-1)
    total_loss = 0
    field_losses = torch.zeros(num_fields)

    for id in range(num_fields):
        field_output = output[:, :, id]
        field_target = target[:, :, id]
        
        field_loss = torch.mean((field_output - field_target) ** 2)
        field_losses[id] = field_loss.item()
        total_loss += field_weights[id] * field_loss

    return total_loss, field_losses

if __name__ == '__main__':

    with open(f'Loss_csv/train_test_loss_{NET_NAME[NET_TYPE]}.csv', 'wt') as fp: 
        pass

    with open('data_split/data_split_Multi_{}.pic'.format(Case_Num), 'rb') as fp: 
        data_split = pickle.load(fp)

        U_train = data_split.U_train.to(device)
        Y_train = data_split.Y_train.to(device)
        G_train = data_split.G_train.to(device)
        # print("U_train.shape, Y_train.shape, G_train.shape", U_train.shape, Y_train.shape, G_train.shape)

        U_test = data_split.U_test.to(device)
        Y_test = data_split.Y_test.to(device)
        G_test = data_split.G_test.to(device)

        n_inputF = U_train.shape[-1]
        n_pointD = Y_train.shape[-1]

        Y_select_indices = torch.randperm(Y_train.size(1))[:N_selected].numpy()
        Yin_train = Y_train[:, Y_select_indices, :].to(device)
        Yin_test = Y_test[:, Y_select_indices, :].to(device)
        print('Y_train.shape = ', Y_train.shape)
        print('Yin_train.shape = ', Yin_train.shape)

        # Extract the temperature values (field_idx = 0) from G_train and G_test or other SELECTED field
        Gin_train = G_train[:, Y_select_indices, field_idx].unsqueeze(-1).to(device)  # Select the first field (temperature)
        Gin_test = G_test[:, Y_select_indices, field_idx].unsqueeze(-1).to(device)  # Select the first field (temperature)
        print('Gin_Train.shape = ', Gin_train.shape)

        # # Save Y_select & the corresponding temperature value to a CSV file
        # with open(f'Y_select/Y_select_Train_{NET_NAME[NET_TYPE]}.csv', 'w', newline='') as csvfile:
        #     csvwriter = csv.writer(csvfile)
        #     csvwriter.writerow(['Case', 'Point Index', 'X Coordinate', 'Y Coordinate', 'Temperature'])
        #     for case in range(Yin_train.size(0)):
        #         for i, point_idx in enumerate(Y_select_indices):  # Removed .cpu().numpy()
        #             csvwriter.writerow([case, point_idx, Yin_train[case, i, 0].item(), Yin_train[case, i, 1].item(), Gin_train[case, i, 0].item()])

        # # Save Y_select & the corresponding temperature value to a CSV file
        # with open(f'Y_select/Y_select_Test_{NET_NAME[NET_TYPE]}.csv', 'w', newline='') as csvfile:
        #     csvwriter = csv.writer(csvfile)
        #     csvwriter.writerow(['Case', 'Point Index', 'X Coordinate', 'Y Coordinate', 'Temperature'])
        #     for case in range(Yin_test.size(0)):
        #         for i, point_idx in enumerate(Y_select_indices):  # Removed .cpu().numpy()
        #             csvwriter.writerow([case, point_idx, Yin_test[case, i, 0].item(), Yin_test[case, i, 1].item(), Gin_test[case, i, 0].item()])

    field_weights = torch.tensor([1.0] * 9)  # Replace with actual weights if needed
    field_weights = field_weights.to(device)

    # Define the network
    if (NET_TYPE == 0):
        net = Direct_SensorToField(n_cond, N_selected, n_baseF).to(device)
    else:
        print('Net is not correctly defined !!!')
        exit()
    # Wrap the model with DataParallel
    net = nn.DataParallel(net, device_ids=device_ids)
    net.apply(weights_init)
    optimizer = optim.Adam(net.parameters(), lr=0.001) 

    # Set up early stopping parameters
    patience = 200
    best_combined_loss = float('5.0') #Initial threshold to determine early stopping
    counter = 0
    train_loss_weight = 0.85
    test_loss_weight = 0.15

    for epoch in range(N_EPOCH):
        start_time = time.time()
        Total_train_loss_Data = 0
        Total_test_loss_Data = 0
        train_batch_count = 0
        test_batch_count = 0

        train_loss = torch.zeros(len(field_names), device=device)  # [len(field_names), len(field_names)]
        test_loss = torch.zeros(len(field_names), device=device)  # [len(field_names), len(field_names)]

        for U, Y, G, Yin, Gin in get_data_iter(U_train, Y_train, G_train, Yin_train, Gin_train):
            optimizer.zero_grad()
            loss = 0.0

            output = net(U, Y, Gin)

            loss_data, losses = custom_mse_loss(output, G, field_weights)
            Total_train_loss_Data += loss_data
            loss += loss_data
            for j, loss_item in enumerate(losses):
                train_loss[j] += loss_item

            loss.backward()
            optimizer.step()
            train_batch_count += 1
        train_loss /= train_batch_count
        Total_train_loss_Data /= train_batch_count

        with torch.no_grad():  # Use no_grad for evaluation in test phase
            for U, Y, G, Yin, Gin in get_data_iter(U_test, Y_test, G_test, Yin_test, Gin_test):
                # print('U.shape is ', U.shape)
                # print(G_max.shape)
                output = net(U, Y, Gin)
                loss_data, losses = custom_mse_loss(output, G, field_weights)
                Total_test_loss_Data += loss_data
                for j, loss_item in enumerate(losses):
                    test_loss[j] += loss_item
                test_batch_count += 1
            test_loss /= test_batch_count
            Total_test_loss_Data /= test_batch_count
        
        combined_loss = train_loss_weight*Total_train_loss_Data + test_loss_weight*Total_test_loss_Data

        end_time = time.time()
        epoch_duration = end_time - start_time

        # Print and write to CSV 
        if (epoch + 1) % 20 == 0:

            print(f'Epoch {epoch+1}/{N_EPOCH}, Duration: {epoch_duration:.4f} seconds')
            print()

            print(f'Epoch {epoch+1}/{N_EPOCH}, Total Train Loss: {Total_train_loss_Data.item()}, Total Test Loss: {Total_test_loss_Data.item()}')
            for id, field_name in enumerate(field_names):
                print(f'Train Loss for field {field_name}: {train_loss[id].item()}, Test Loss for field {field_name}: {test_loss[id].item()}')

            # Write to CSV file
            with open(f'Loss_csv/train_test_loss_{NET_NAME[NET_TYPE]}.csv', 'at', newline='') as fp:
                writer = csv.writer(fp, delimiter='\t')
                if ((epoch + 1) // 20 == 1):
                    fp.write(NET_SETTINGS)
                    header = ['Epoch', 'Overall_Train_Loss', 'Overall_Test_Loss']
                    interleaved_field_names = [fn for field_name in field_names for fn in (f'Train_{field_name}_loss', f'Test_{field_name}_loss')]
                    header.extend(interleaved_field_names)
                    writer.writerow(header)
                row_data = [epoch + 1, Total_train_loss_Data.item(), Total_test_loss_Data.item()] 
                # Combine train and test losses for each field into pairs
                for train_loss, test_loss in zip(train_loss, test_loss):
                    row_data.append(f'{train_loss.item()}')
                    row_data.append(f'{test_loss.item()}')
                writer.writerow(row_data)

            # Determine whether should trigger early stopping every 200 epochs
            if ( combined_loss < best_combined_loss ):
                best_combined_loss = combined_loss
                print(f'Best combined loss so far is {best_combined_loss}, still improving')
                counter = 0
                
                with open('Output_Net/net_MultiField_{}.pic'.format(NET_NAME[NET_TYPE]), 'wb') as fp: #每隔一段时间更新网络 !!文件名需要记得修改!!
                    pickle.dump(net, fp)
                    print('...成功保存 net.pic')
                    print()

            else:
                counter += 1
                print(f'Best combined loss so far is {best_combined_loss}, NOT further improving')
                print(f'The counter for triggering early stopping is {counter}')
                print()
                if counter >= patience:
                    print("Early stopping triggered")
                    break

        torch.cuda.empty_cache()