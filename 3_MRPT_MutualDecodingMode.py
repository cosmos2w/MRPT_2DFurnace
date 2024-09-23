
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
from network import MRPT_PreTrain_Net_MutualDecodingMode

# Specify the GPUs to use
device_ids = [0, 1]
device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

WRITE_FEATURE = False   # Determine whether feature vectors should be documented

N_EPOCH = 2000000
Case_Num = 300
n_field_info = 36
n_baseF = 50 
field_names = ['T', 'P', 'Vx', 'Vy', 'O2', 'CO2', 'H2O', 'CO', 'H2']
Unified_Weight = 5.0 # Contribution of the unified feature
N_P_Selected = 200

#Transformer layer parameters
num_heads = 6
num_layers = 1

NET_SETTINGS = f'n_field_info = {n_field_info}\tMultiHeadAttention={num_heads} & layer=1\tn_baseF = {n_baseF}\tnet_Y_Gin=[n_baseF + 1, 50, 50, n_field_info]\tConNet=[n_field_info, 50, 50, n_base]\tPositionNet([2, 60, 60, 60, n_base])\n'
NET_NAME = f'MRPT_MutualDecoding_{N_P_Selected}'

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

def custom_mse_loss(output, target, field_weights):
    # assert output.shape == target.shape, "Output and target must have the same shape"

    num_fields = target.size(-1)
    total_loss = 0
    weighted_total_loss = 0
    field_losses = torch.zeros(num_fields)

    for field_idx in range(num_fields):
        field_output = output[:, :, field_idx]
        field_target = target[:, :, field_idx]
        field_loss = torch.mean((field_output - field_target) ** 2)
        
        field_losses[field_idx] = field_loss.item()
        total_loss += field_loss
        weighted_total_loss += field_weights[field_idx] * field_loss

    # assert len(field_names) == len(losses), "The length of field_names must match the number of fields in field_losses"
    return weighted_total_loss, total_loss, field_losses

if __name__ == '__main__':

    for field_idx in range(len(field_names) + 1):
        field_name_or_feature = 'Unified_feature' if field_idx == len(field_names) else field_names[field_idx]
        with open(f'Loss_csv/train_test_loss_Mutual/train_test_loss_MultiField_{NET_NAME}_{field_name_or_feature}.csv', 'wt') as fp: 
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

    field_weights_default = torch.tensor([1.0] * 9)  # Replace with actual weights if needed
    field_weights_default = field_weights_default.to(device)

    Ini_Concern_weights = torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
    Ini_Concern_weights = Ini_Concern_weights.to(device)

    net = MRPT_PreTrain_Net_MutualDecodingMode(n_field_info, n_baseF, num_heads, num_layers, num_fields=len(field_names)).to(device)
    net = nn.DataParallel(net, device_ids=device_ids)
    net.apply(weights_init)

    optimizer = optim.Adam(net.parameters(), lr=0.00050)  # weight_decay=1e-4
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.98, patience=1000, verbose=True, min_lr=0.0002)

    # Set up early stopping parameters
    patience = 500
    best_combined_loss = float('50.0') #Initial threshold to determine early stopping
    counter = 0
    train_loss_weight = 0.25
    test_loss_weight = 0.75

    Concern_weights = torch.zeros(len(field_names))
    Concern_weights = Ini_Concern_weights

    Field_weights_All = torch.zeros(len(field_names), len(field_names))
    Field_weights_All = Field_weights_All.fill_(1.0).to(device)

    for epoch in range(N_EPOCH):

        Total_train_loss_Data = 0 # the total losses from all generated fields from each field: Loss_T = Sigma(i) Sigma(j) Field_j(Generated from Field_i)
        Total_test_loss_Data = 0
        train_loss = torch.zeros(len(field_names)+1, device=device)  # [len(field_names), len(field_names)], add the contribution from the unified_feature
        test_loss = torch.zeros(len(field_names)+1, device=device)  # [len(field_names), len(field_names)]

        train_CS_loss = torch.zeros(len(field_names)+1, device=device)  # [len(field_names), len(field_names)], add the contribution from the unified_feature
        test_CS_loss = torch.zeros(len(field_names)+1, device=device)  # [len(field_names), len(field_names)]

        # Initialize a two-dimensional tensor for training losses
        train_losses = torch.zeros(len(field_names)+1, len(field_names), device=device)  # [len(field_names), len(field_names)]
        test_losses = torch.zeros(len(field_names)+1, len(field_names), device=device)  # [len(field_names), len(field_names)]

        train_batch_count = 0
        test_batch_count = 0

        for U, Y, G in get_data_iter(U_train, Y_train, G_train, N = N_P_Selected):

            optimizer.zero_grad()
            output = net(U, Y, G, num_heads)
            output_stacked = torch.stack(output, dim=0)
            loss = 0
            
            for field_idx in range(len(field_names) + 1):  # Assuming the last index is for the unified feature
                output_Select = output_stacked[field_idx, :, :, :]
                field_weights = Field_weights_All[field_idx, :] if field_idx < len(field_names) else field_weights_default
                concern_weight = Concern_weights[field_idx] if field_idx < len(field_names) else Unified_Weight

                weighted_loss_data, loss_data, losses = custom_mse_loss(output_Select, G, field_weights)

                loss += weighted_loss_data * concern_weight
                            
                Total_train_loss_Data += loss_data
                train_loss[field_idx] += loss_data
                # Update train_losses for the current field or unified feature
                for j, loss_item in enumerate(losses):
                    train_losses[field_idx, j] += loss_item

            train_batch_count += 1
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

        train_loss /= train_batch_count
        train_CS_loss /= train_batch_count
        train_losses /= train_batch_count
        Total_train_loss_Data /= train_batch_count

        with torch.no_grad(): 
            # Normalize weights for other fields based on the first field's loss
            for field_idx in range(0, len(field_names)):
                Concern_weights[field_idx] = (train_loss[0] / train_loss[field_idx]) ** 1.0

                for id in range(0, len(field_names)):
                    slice_min, min_idx = torch.min(train_losses[0:9, id], dim=0) # Find the min train loss for generated id(th) field from all fields
                    Field_weights_All[field_idx, id] = (slice_min / train_losses[field_idx, id]) ** 2.0
                Field_weights_All[field_idx, field_idx] = 1.0 # For each field, allow it to fully focus on itself

        # Print and write to CSV every 20 epochs
        if (epoch + 1) % 20 == 0:

            with torch.no_grad():  # Make sure  to use no_grad for evaluation in test phase

                for U, Y, G in get_data_iter(U_test, Y_test, G_test, N = 2000):
                    output = net(U, Y, G, num_heads)
                    output_stacked = torch.stack(output, dim=0)
                    for field_idx in range(len(field_names) + 1):
                        output_Select = output_stacked[field_idx, :, :, :]
                        weighted_loss_data, loss_data, losses = custom_mse_loss(output_Select, G, field_weights)

                        Total_test_loss_Data += loss_data
                        test_loss[field_idx] += loss_data
                        
                        for j, loss in enumerate(losses):
                            test_losses[field_idx, j] += loss

                    test_batch_count += 1

            test_loss /= test_batch_count
            test_CS_loss /= test_batch_count
            test_losses /= test_batch_count
            Total_test_loss_Data /= test_batch_count

            combined_loss = train_loss_weight*train_loss[len(field_names)] + test_loss_weight*test_loss[len(field_names)]

            print(f'Epoch {epoch+1}/{N_EPOCH}, Total Train Loss: {Total_train_loss_Data.item()}, Total Test Loss: {Total_test_loss_Data.item()}')
            for field_idx, field_name in enumerate(field_names):
                print(f'Sub-total Train Loss for {field_name}: {train_loss[field_idx].item()}, Sub-total Test Loss for {field_name}: {test_loss[field_idx].item()}')
            print(f'Sub-total Train Loss for Unified feature: {train_loss[len(field_names)].item()}, Sub-total Test Loss for Unified feature: {test_loss[len(field_names)].item()}\n')

            # Write to CSV file
            for field_idx in range(len(field_names) + 1):
                # Determine field name or unified feature
                field_name_or_feature = 'Unified_feature' if field_idx == len(field_names) else field_names[field_idx]
                with open(f'Loss_csv/train_test_loss_Mutual/train_test_loss_MultiField_{NET_NAME}_{field_name_or_feature}.csv', 'at', newline='') as fp:
                    writer = csv.writer(fp, delimiter='\t')
                    if ((epoch + 1) // 20 == 1):
                        fp.write(NET_SETTINGS)

                        header = ['Epoch', 'Overall_Train_Loss', 'Overall_Test_Loss']
                        interleaved_field_names = [fn for name in field_names + ['Unified_feature'] for fn in (f'Train_{name}_loss', f'Test_{name}_loss')]
                        header.extend(interleaved_field_names)
                        writer.writerow(header)

                    row_data = [epoch + 1, train_loss[field_idx].item(), test_loss[field_idx].item()] + \
                            [item for pair in zip(train_losses[field_idx].tolist(), test_losses[field_idx].tolist()) for item in pair]
                    writer.writerow(row_data)

            # Determine whether should trigger early stopping every 200 epochs
            if ( combined_loss < best_combined_loss ):
                best_combined_loss = combined_loss
                print(f'Best combined loss so far is {best_combined_loss}, still improving')
                counter = 0

                model_save_path = 'Output_Net/net_{}_state_dict.pth'.format(NET_NAME)
                torch.save(net.module.state_dict(), model_save_path)
                print('Successfully saved the latest best net at {}'.format(model_save_path))

                if (WRITE_FEATURE == True):
                    with open(f'LatentRepresentation/U_FieldInfoSensor_{NET_NAME}.csv', 'w', newline='') as csvfile:
                        fieldnames = ['U', 'Unified_field_info']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        # Write data from the training set
                        writer.writerow({'U': 'From training set:'})
                        with torch.no_grad():
                            for U_batch, Yin_batch, Gin_batch in get_data_iter(U_train, Y_train, G_train, N_P_Selected = 200):
                                field_info = net.module._compress_data(Yin_batch, Gin_batch, num_heads)
                                U_Unified = net.module.attention(field_info)
                                # U_Unified = net.module.mlp(U_Unified)
                                for i in range(U_batch.shape[0]):
                                    row = {'U': U_batch[i].cpu().numpy().tolist(),
                                        'Unified_field_info': U_Unified[i].cpu().numpy().tolist()}
                                    writer.writerow(row)
                        # Write data from the test set
                        writer.writerow({'U': 'From test set:'})
                        with torch.no_grad():
                            for U_batch, Yin_batch, Gin_batch in get_data_iter(U_test, Y_test, G_test, N_P_Selected = 200):
                                field_info = net.module._compress_data(Yin_batch, Gin_batch, num_heads)
                                U_Unified = net.module.attention(field_info)
                                # U_Unified = net.module.mlp(U_Unified)
                                for i in range(U_batch.shape[0]):
                                    row = {'U': U_batch[i].cpu().numpy().tolist(),
                                        'Unified_field_info': U_Unified[i].cpu().numpy().tolist()}
                                    writer.writerow(row)
            else:
                counter += 1
                print(f'Best combined loss so far is {best_combined_loss}, NOT further improving')
                print(f'The counter for triggering early stopping is {counter}')
                if counter >= patience:
                    print("Early stopping triggered")
                    break
