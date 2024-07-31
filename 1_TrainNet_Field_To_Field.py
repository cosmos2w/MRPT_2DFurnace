
import sys
sys.path.append('..')

import io
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

from torch import nn 
from constant import DataSplit 
from network import FieldToField_TransformerNet

torch.cuda.set_device(1)

# Specify the GPUs to use
device_ids = [1, 2]
device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

#__________________________PARAMETERS_________________________________
N_EPOCH = 2000000
Case_Num = 300
n_field_info = 36
n_baseF = 40 
N_P_Selected = 1000

num_heads = 9
num_layers = 1

field_names = ['T', 'P', 'Vx', 'Vy', 'O2', 'CO2', 'H2O', 'CO', 'H2']
skip_field = 1      
#____________________________________________________________________

NET_SETTINGS = f'Case_Num = {Case_Num}\tn_field_info = {n_field_info}\tn_baseF = {n_baseF}\tnet_Y_Gin=[ n_base + 1 , 60, 60, n_field_info]\tConNet=[n_field_info, 50, 50, n_field_info]\n'
NET_NAME = f'F2F_Input_{field_names[skip_field]}'

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

def get_data_iter(U, Y, G, N_Point, batch_size = 360): # random sampling in each epoch
    num_examples = len(U)
    num_points = Y.shape[1]
    indices = list(range(num_examples))
    np.random.shuffle(indices) 
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) 
        j = j.to(device)

        selected_points = torch.randperm(num_points)[:N_Point].to(device)
        yield  U.index_select(0, j), Y.index_select(0, j).index_select(1, selected_points), G.index_select(0, j).index_select(1, selected_points)

def custom_mse_loss(output, target, field_weights=None):
    # output and target have shape [batch_size, n_points, n_fields]
    assert output.shape == target.shape, "Output and target must have the same shape"
    losses = []
    for k in range(output.shape[2]):  # Loop over fields
        field_loss = F.mse_loss(output[:, :, k], target[:, :, k], reduction='mean')
        if field_weights is not None:
            field_loss *= field_weights[k]
        losses.append(field_loss)
    total_loss = torch.stack(losses).sum()

    assert len(field_names) == len(losses), "The length of field_names must match the number of fields in field_losses"
    
    return total_loss, losses

if __name__ == '__main__':

    with open('Loss_csv/train_test_loss_{}.csv'.format(NET_NAME), 'wt') as fp: 
        pass

    with open('data_split/data_split_Multi_{}.pic'.format(Case_Num), 'rb') as fp: 
        data_split = pickle.load(fp)

        U_train = data_split.U_train.to(device)
        Y_train = data_split.Y_train.to(device)
        G_train = data_split.G_train.to(device)

        U_test = data_split.U_test.to(device)
        Y_test = data_split.Y_test.to(device)
        G_test = data_split.G_test.to(device)

        n_inputF = U_train.shape[-1]
        n_pointD = Y_train.shape[-1]

    field_weights = torch.tensor([1.0] * 9)  # Replace with actual weights if needed
    field_weights = field_weights.to(device)

    net = FieldToField_TransformerNet(n_field_info, n_baseF, num_heads, num_layers).to(device)
    net = nn.DataParallel(net, device_ids=device_ids)
    net.apply(weights_init)
    
    optimizer = optim.Adam(net.parameters(), lr=0.00050)  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=1000, verbose=True, min_lr=0.00020)

    # Set up early stopping parameters
    patience = 200
    best_combined_loss = float('0.5') #Initial threshold to determine early stopping
    counter = 0
    train_loss_weight = 0.25
    test_loss_weight = 0.75

    for epoch in range(N_EPOCH): 

        Total_train_loss = 0 # the total losses from all generated fields from each field: Loss_T = Sigma(i) Sigma(j) Field_j(Generated from Field_i)
        Total_test_loss = 0
        train_losses = torch.zeros(len(field_names), device=device)  # [len(field_names), len(field_names)]
        test_losses = torch.zeros(len(field_names), device=device)  # [len(field_names), len(field_names)]

        train_batch_count = 0
        test_batch_count = 0

        for U, Y, G in get_data_iter(U_train, Y_train, G_train, N_Point = N_P_Selected):
            loss = 0
            optimizer.zero_grad()

            G_in = G[:, :, [skip_field]]
            Gout = net(Y, G_in, num_heads)
            loss_data, losses = custom_mse_loss(Gout, G, field_weights)

            loss = loss_data
            Total_train_loss = loss_data
            for j, loss_item in enumerate(losses):
                train_losses[j] += loss_item
            
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            train_batch_count += 1

        train_losses /= train_batch_count
        Total_train_loss /= train_batch_count

        if (epoch+1) % 20 ==0:

            with torch.no_grad():  
                for U, Y, G in get_data_iter(U_test, Y_test, G_test, N_Point = 2000):

                    G_in = G[:, :, [skip_field]]
                    Gout = net(Y, G_in, num_heads)
                    loss_data, losses = custom_mse_loss(Gout, G, field_weights)

                    Total_test_loss = loss_data
                    for j, loss_item in enumerate(losses):
                        test_losses[j] += loss_item
                    
                    test_batch_count += 1

                test_losses /= test_batch_count
                Total_test_loss /= test_batch_count

            print(f'\nEpoch {epoch+1}/{N_EPOCH}, Total train_Loss: {Total_train_loss.item()}, Total test_Loss: {Total_test_loss.item()} ')
            for field_name, train_loss, test_loss in zip(field_names, train_losses, test_losses):
                print(f'Train Loss for field {field_name}: {train_loss.item()}, Test Loss for field {field_name}: {test_loss.item()}')
            
            combined_loss = train_loss_weight*Total_train_loss + test_loss_weight*Total_test_loss

            with open('Loss_csv/train_test_loss_{}.csv'.format(NET_NAME), 'at') as fp: 
                if ((epoch + 1) / 20 == 1): 
                    fp.write(NET_SETTINGS)
                    row_data = [f'Epoch',f'train_loss_Data',f'test_loss_Data']
                    for field_name in (field_names):
                        row_data.append(f'Train_{field_name}_loss')
                        row_data.append(f'Test_{field_name}_loss')

                    fp.write('\t'.join(row_data) + '\n')

                row_data = [f'{epoch+1}', f'{Total_train_loss.item()}', f'{Total_test_loss.item()}']
                # Combine train and test losses for each field into pairs
                for train_loss, test_loss in zip(train_losses, test_losses):
                    row_data.append(f'{train_loss.item()}')
                    row_data.append(f'{test_loss.item()}')
                fp.write('\t'.join(row_data) + '\n')
                pass
            
            # Determine whether should trigger early stopping
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
                print(f'The counter for triggering early stopping is {counter}')
                if counter >= patience:
                    print("Early stopping triggered")
                    break



