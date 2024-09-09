
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

from torch import nn 
from constant import DataSplit 
from network import Mutual_Representation_PreTrain_Net

torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Specify the GPUs to use
device_ids = [1, 2]
device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

#__________________________PARAMETERS_________________________________
N_EPOCH = 200000
Case_Num = 300
N_P_Selected = 200
n_field_info = 36
n_baseF = 50 
field_names = ['T', 'P', 'Vx', 'Vy', 'O2', 'CO2', 'H2O', 'CO', 'H2']

Unified_Weight = 5.0 # Contribution of the unified feature

#Transformer layer parameters
num_heads = 6
num_layers = 1

RELOAD = False # To determine if the training starts from a previous aborted task
#____________________________________________________________________

NET_SETTINGS = 'Field_weights [^2.0] & Unified 5.0, use drop-out 0.50 & 0.10 & 0.10\tUnified_Weight = 5.0\tn_field_info = 36\tMultiHeadAttention={num_heads} & layer=1\tn_baseF = 40\tnet_Y_Gin=[n_baseF + 1, 50, 50, n_field_info]\tConNet=[n_field_info, 50, 50, n_base]\tPositionNet([2, 50, 50, 50, n_base])\n'
NET_NAME = f'MRPT_Standard_{N_P_Selected}_2'

Reload_Net_Name = 'net_MRPT_Standard_200_state_dict'
Reload_file_path = 'Output_Net/{}.pth'.format(Reload_Net_Name)

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def get_data_iter(U, Y, G, N, batch_size = 100): # random sampling in each epoch
    num_examples = len(U)
    num_points = Y.shape[1]
    indices = list(range(num_examples))
    np.random.shuffle(indices)  
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) 
        j = j.to(device)

        selected_points = torch.randperm(num_points)[:N].to(device)
        yield  U.index_select(0, j), Y.index_select(0, j).index_select(1, selected_points), G.index_select(0, j).index_select(1, selected_points)

def field_custom_mse_loss(output, target, field_idx, field_weights):

    # print("output.device is", output.device)
    # print("target.device is", target.device)

    num_fields = target.size(-1)
    total_loss = 0
    weighted_total_loss = 0
    field_losses = torch.zeros(num_fields+1)#.to(device)

    for id in range(num_fields + 1):
        field_output = output[:, :, id] #   The reconstructed field_idx(th) field from id(th) field or Unified
        field_target = target[:, :, field_idx]  # Ground truth of the field_idx(th) field
        field_loss = torch.mean((field_output - field_target) ** 2)
        
        field_losses[id] = field_loss.item()
        total_loss += field_loss
        
        field_weight = field_weights[id] if id < len(field_names) else 5.0
        weighted_total_loss += field_weight * field_loss
    
    sorted_losses, sorted_indices = torch.sort(field_losses[:num_fields])

    return weighted_total_loss, total_loss, field_losses, sorted_indices.tolist()

def custom_mse_loss(output, target):
    # assert output.shape == target.shape, "Output and target must have the same shape"

    num_fields = target.size(-1)
    total_loss = 0
    field_losses = torch.zeros(num_fields)

    for field_idx in range(num_fields):
        field_output = output[:, :, field_idx]
        field_target = target[:, :, field_idx]
        field_loss = torch.mean((field_output - field_target) ** 2)
        
        field_losses[field_idx] = field_loss.item()
        total_loss += field_loss

    # assert len(field_names) == len(losses), "The length of field_names must match the number of fields in field_losses"
    return total_loss, field_losses

if __name__ == '__main__':

    for field_idx in range(len(field_names)):   # Saving losses to reconstruct the field_idx(th) field from [all fields & Unifed]
        with open(f'Loss_csv/train_test_loss_Mutual/train_test_loss_{NET_NAME}_For_{field_names[field_idx]}.csv', 'wt') as fp:
            pass
    with open(f'Loss_csv/train_test_loss_Mutual/train_test_loss_{NET_NAME}_GlobalUnifed.csv', 'wt') as fp:
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

    field_weights_default = torch.tensor([1.0] * 9)  # Replace with actual weights if needed
    field_weights_default = field_weights_default.to(device)

    net = Mutual_Representation_PreTrain_Net(n_field_info, n_baseF, num_heads, num_layers, num_fields=len(field_names)).to(device)

    if RELOAD is True:
        state_dict = torch.load(Reload_file_path)
        net.load_state_dict(state_dict)
        print(f'\nI have Loaded the pretrained net from {Reload_file_path}\n')

    # Wrap the model with DataParallel 
    net = nn.DataParallel(net, device_ids=device_ids)
    if RELOAD is False: net.apply(weights_init)

    optimizer = optim.Adam(net.parameters(), lr=0.00050)  # , weight_decay=1e-5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.98, patience=2000, verbose=True, min_lr=0.0002)

    # Set up early stopping parameters
    patience = 500
    best_combined_loss = float('5.0') #Initial threshold to determine early stopping
    counter = 0
    train_loss_weight = 0.15
    test_loss_weight = 0.85

    Field_weights_All = torch.zeros(len(field_names), len(field_names))
    Field_weights_All = Field_weights_All.fill_(1.0).to(device)


    for epoch in range(N_EPOCH):

        Total_train_loss_Data = 0 # the total losses from all generated fields from each field: Loss_T = Sigma(i) Sigma(j) Field_j(Generated from Field_i)
        Total_test_loss_Data = 0
        train_loss = torch.zeros(len(field_names), device=device)  
        test_loss = torch.zeros(len(field_names), device=device)  

        Unified_train_loss = 0 
        Unified_test_loss = 0

        Unified_train_losses = torch.zeros(len(field_names), device=device)  
        Unified_test_losses = torch.zeros(len(field_names), device=device)  

        # Initialize a two-dimensional tensor for training losses
        train_losses = torch.zeros(len(field_names), len(field_names)+1, device=device)  
        test_losses = torch.zeros(len(field_names), len(field_names)+1, device=device)  

        train_batch_count = 0
        test_batch_count = 0

        for U, Y, G in get_data_iter(U_train, Y_train, G_train, N = N_P_Selected):
            optimizer.zero_grad()

            output, Unified_output = net(U, Y, G, num_heads)
            output_stacked = torch.stack(output, dim=0) 
            # print("output_stacked shape:", output_stacked.shape)  # Should print [n_fields (from), n_batch, n_point_select, n_fields+1 (for, plus a Unifed)]
            loss = 0
            
            for field_idx in range(len(field_names)):  # Calculating the losses to reconstruct field_idx(th) field from all fields
                
                output_Select = output_stacked[field_idx, :, :, :] # Picking out the reconstructed field_idx(th) field results from all fields & Unified
                
                field_weights = Field_weights_All[field_idx, :]
                weighted_loss_data, loss_data, losses, sorted_indices = field_custom_mse_loss(output_Select, G, field_idx, field_weights)

                loss += weighted_loss_data
                               
                Total_train_loss_Data += loss_data
                train_loss[field_idx] += loss_data
                # Update train_losses for the current field or unified feature
                losses = losses.to(device)
                # print("train_losses.device is", train_losses.device)
                # print("losses.device is", losses.device)
                for j, loss_item in enumerate(losses):
                    train_losses[field_idx, j] += loss_item

            # Losses for reconstructing all fields from the Global Unified representation
            Unifed_loss_data, Unifed_losses = custom_mse_loss(Unified_output, G)
            Unifed_losses = Unifed_losses.to(device)
            loss += Unifed_loss_data * Unified_Weight
            Total_train_loss_Data += Unifed_loss_data
            Unified_train_loss += Unifed_loss_data
            for j, loss_item in enumerate(Unifed_losses):
                Unified_train_losses[j] += loss_item

            train_batch_count += 1
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

        train_loss /= train_batch_count
        train_losses /= train_batch_count
        Unified_train_loss /= train_batch_count
        Unified_train_losses /= train_batch_count
        Total_train_loss_Data /= train_batch_count

        with torch.no_grad(): 
            # Normalize weights for other fields based on the first field's loss
            #   Field_weights_All[i, j] means: the weight in field_custom_mse_loss for the i(th) output field from the j(th) input field
            for field_idx in range(0, len(field_names)): #  field_idx is the index for output field
                for id in range(0, len(field_names)):   #   id is the index for input field
                    slice_min, min_idx = torch.min(train_losses[field_idx, 0:9], dim=0) # Find the min train loss for the field_idx(th) output field from all fields
                    # print(f"Minimum value for field {id}: {slice_min}, at index: {min_idx}, current {field_idx} is {train_losses[field_idx, id]}")
                    Field_weights_All[field_idx, id] = (slice_min / train_losses[field_idx, id]) ** 2.0
                Field_weights_All[field_idx, field_idx] = 1.0 # For each field, allow it to fully focus on itself

        PRINT_NUM = int(20) 
        if (epoch + 1) % PRINT_NUM == 0:

            with torch.no_grad():  # Make sure  to use no_grad for evaluation in test phase

                for U, Y, G in get_data_iter(U_test, Y_test, G_test, N = 2000):
                    # output, Unified_output = net(U, Y, G, att_index_expanded, num_heads)
                    output, Unified_output = net(U, Y, G, num_heads)
                    output_stacked = torch.stack(output, dim=0) 
                    
                    for field_idx in range(len(field_names)):  # Calculating the losses to reconstruct field_idx(th) field from all fields
                        
                        output_Select = output_stacked[field_idx, :, :, :] # Picking out the reconstructed field_idx(th) field results from all fields & Unified                        
                        field_weights = Field_weights_All[field_idx, :] 
                        weighted_loss_data, loss_data, losses, sorted_indices = field_custom_mse_loss(output_Select, G, field_idx, field_weights)  
                                        
                        Total_test_loss_Data += loss_data
                        test_loss[field_idx] += loss_data
                        # Update train_losses for the current field or unified feature
                        losses = losses.to(device)
                        for j, loss_item in enumerate(losses):
                            test_losses[field_idx, j] += loss_item

                    # Losses for reconstructing all fields from the Global Unified representation
                    Unifed_loss_data, Unifed_losses = custom_mse_loss(Unified_output, G)
                    Unifed_losses = Unifed_losses.to(device)
                    Total_test_loss_Data += Unifed_loss_data
                    Unified_test_loss += Unifed_loss_data
                    for j, loss_item in enumerate(Unifed_losses):
                        Unified_test_losses[j] += loss_item

                    test_batch_count += 1
            
            test_loss /= test_batch_count
            test_losses /= test_batch_count
            Unified_test_loss /= test_batch_count
            Unified_test_losses /= test_batch_count
            Total_test_loss_Data /= test_batch_count
            
            combined_loss = train_loss_weight*Unified_train_loss + test_loss_weight*Unified_test_loss

            print(f'Epoch {epoch+1}/{N_EPOCH}, Total Train Loss: {Total_train_loss_Data.item()}; Total Test Loss: {Total_test_loss_Data.item()}')
            for field_idx, field_name in enumerate(field_names):
                print(f'Sub-total Train Loss for {field_name}: {train_loss[field_idx].item()}; Sub-total Test Loss for {field_name}: {test_loss[field_idx].item()}')
                print(f'Train Loss for {field_name} from {field_name}: {train_losses[field_idx, field_idx].item()}; Test Loss for {field_name} from {field_name}: {test_losses[field_idx, field_idx].item()}')
                print(f'Train Loss for {field_name} from Unified: {train_losses[field_idx, len(field_names)].item()}; Test Loss for {field_name} from Unified: {test_losses[field_idx, len(field_names)].item()}')
                print()
            
            print(f'Total Train Loss for Unified feature: {Unified_train_loss.item()}; Total Test Loss for Unified feature: {Unified_test_loss.item()}')
            for field_idx, field_name in enumerate(field_names):
                print(f'Unifed Train Loss for {field_name}: {Unified_train_losses[field_idx].item()}; Unified Test Loss for {field_name}: {Unified_test_losses[field_idx].item()}')

            # Write to CSV file
            for field_idx in range(len(field_names)):
                # Determine field name or unified feature
                with open(f'Loss_csv/train_test_loss_Mutual/train_test_loss_{NET_NAME}_For_{field_names[field_idx]}.csv', 'at', newline='') as fp:
                    writer = csv.writer(fp, delimiter='\t')
                    if ((epoch + 1) // PRINT_NUM == 1):
                        fp.write(NET_SETTINGS)
                        header = ['Epoch', 'Overall_Train_Loss', 'Overall_Test_Loss']
                        interleaved_field_names = [fn for name in field_names + ['Unified_feature'] for fn in (f'Train_{name}_loss', f'Test_{name}_loss')]
                        header.extend(interleaved_field_names)
                        writer.writerow(header)

                    row_data = [epoch + 1, train_loss[field_idx].item(), test_loss[field_idx].item()] + \
                            [item for pair in zip(train_losses[field_idx].tolist(), test_losses[field_idx].tolist()) for item in pair]
                    writer.writerow(row_data)
            with open(f'Loss_csv/train_test_loss_Mutual/train_test_loss_{NET_NAME}_GlobalUnifed.csv', 'at', newline='') as fp:
                writer = csv.writer(fp, delimiter='\t')
                if ((epoch + 1) // PRINT_NUM == 1):
                    fp.write(NET_SETTINGS)
                    header = ['Epoch', 'Overall_Train_Loss', 'Overall_Test_Loss']
                    interleaved_field_names = [fn for name in field_names for fn in (f'Train_{name}_loss', f'Test_{name}_loss')]
                    interleaved_field_names.append('att_index')
                    header.extend(interleaved_field_names)
                    writer.writerow(header)

                row_data = [epoch + 1, Unified_train_loss.item(), Unified_test_loss.item()] + \
                        [item for pair in zip(Unified_train_losses.tolist(), Unified_test_losses.tolist()) for item in pair] 
                writer.writerow(row_data)

            # Determine whether should trigger early stopping every 200 epochs
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
