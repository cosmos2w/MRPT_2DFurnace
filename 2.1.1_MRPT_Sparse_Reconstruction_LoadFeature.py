
#__________________________________________________________________________________________________________________________________________
# In this version, the code will load the pretrained net
# Downstream tasks; Generate random sparse sensor points from the Data_split, recover the unified latent feature using a MLP, and then rebuild all the fields
# The pre-trained features will be initially loaded and exported so that in sparse reconstruction phase there's no need to obtained the features from encoder in each epoch
#__________________________________________________________________________________________________________________________________________

import sys
sys.path.append('..')

import io
import csv
import time
import numpy as np 

import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from torch import nn 
from constant import DataSplit 
from network import Direct_SensorToFeature, Mutual_Representation_PreTrain_Net, Mutual_SensorToFeature_InterInference, Mutual_SensorToFeature_InterInference_LoadFeature

# Specify the GPUs to use
device_ids = [1]
device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

#__________________________PARAMETERS_________________________________
field_names = ['T', 'P', 'Vx', 'Vy', 'O2', 'CO2', 'H2O', 'CO', 'H2']
n_fields    = len(field_names)
N_EPOCH     = 1000000
N_REPORT    = 10
Case_Num    = 300

EVALUATE     = True
n_field_info = 36
n_baseF      = 40 
n_cond       = 11 #Length of the condition vector in U

field_idx    = 0 # The field used for sparse reconstruction
N_selected   = 50 # Points selected for sparse reconstruction
N_P_Selected = 2000 # points to evaluate performances
#Transformer layer parameters
num_heads    = 9
num_layers   = 1
hidden_sizes = [1000, 1000]
layer_sizes = [n_field_info * len(field_names)] + hidden_sizes + [n_field_info]  
Final_layer_sizes = [n_field_info] + hidden_sizes + [n_field_info] 

Unifed_weight = 5.0

NET_TYPE = int(0) 
                # 0 = [Mutual_Representation_PreTrain_Net]; 1 = [Direct_SensorToFeature]
NET_SETTINGS = f'LR = 5E-4, weight_decay = 5.0E-5\tSelecting {N_selected} random sensors from {field_names[field_idx]} to recover the latent features\thidden_sizes = {hidden_sizes}\n'
NET_NAME = [f'MRPT_SensorToFeature_N{N_selected}_LoadFeature', f'Direct_SensorToFeature_N{N_selected}']

PreTrained_Net_Name = 'net_MRPT_Standard_state_dict'
Load_file_path = 'Output_Net/{}.pth'.format(PreTrained_Net_Name)

def get_data_iter(U, Y, G, Yin, Gin, batch_size = 360): # random sampling in each epoch
    num_examples = len(U)
    num_points = Y.shape[1]
    indices = list(range(num_examples))
    # np.random.shuffle(indices) 
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) 
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

def normalize_data(data, min_val, max_val):
    # Normalize data to [0, 1]
    normalized_data = (data - min_val) / (max_val - min_val)
    # Take the square root of the normalized data
    sqrt_normalized_data = torch.sqrt(normalized_data)
    return sqrt_normalized_data

def Denormalize_data(normalized_data, min_val, max_val):

    Retrieved_data = normalized_data ** 2
    Retrieved_data = Retrieved_data * (max_val - min_val) + min_val
    return Retrieved_data

if __name__ == '__main__':

    with open(f'Loss_csv/train_test_loss_reconstruction/train_test_loss_S2F_{NET_NAME[NET_TYPE]}.csv', 'wt') as fp: 
        pass

    if EVALUATE is True:
        with open(f'Loss_csv/train_test_loss_reconstruction/Reconstruct_loss_{NET_NAME[NET_TYPE]}.csv', 'wt') as fp:
            pass
    #     with open(f'LatentRepresentation/predictions_{NET_NAME[NET_TYPE]}.csv', 'wt') as fp:
    #         pass

    # Load the pretrained net
    PreTrained_net = Mutual_Representation_PreTrain_Net(n_field_info, n_baseF, num_heads, num_layers, num_fields=len(field_names)).to(device)
    state_dict = torch.load(Load_file_path)
    PreTrained_net.load_state_dict(state_dict)

    #   Extract all the min and max values in latent features for nor- and denormalization
    Max_Value_Train_list = []
    Min_Value_Train_list = []
    Max_Value_Test_list = []
    Min_Value_Test_list = []
    for id in range(len(field_names) + 1):
        field_name_or_feature = 'Unified' if id == len(field_names) else field_names[id]
        file_path = f'LatentRepresentation/FeaturesFrom_{PreTrained_Net_Name}_To_{field_name_or_feature}.csv'
        
        # Open the CSV file for reading
        try:
            with open(file_path, 'rt') as fp:
                reader = csv.reader(fp)
                next(reader)  # Skip the first row (header)
                second_row = next(reader)  # Read the second row
                
                # Append each value from the second row to the respective list
                Max_Value_Train_list.append(float(second_row[0]))
                Min_Value_Train_list.append(float(second_row[1]))
                Max_Value_Test_list.append(float(second_row[2]))
                Min_Value_Test_list.append(float(second_row[3]))
        except Exception as e:
            print(f"Failed to process {file_path}: {str(e)}")
    # Convert lists to tensors
    Max_Value_Train = torch.tensor(Max_Value_Train_list)
    Min_Value_Train = torch.tensor(Min_Value_Train_list)
    Max_Value_Test = torch.tensor(Max_Value_Test_list)
    Min_Value_Test = torch.tensor(Min_Value_Test_list)

    print(f"Max_Value_Train: {Max_Value_Train}")
    print(f"Min_Value_Train: {Min_Value_Train}")
    print(f"Max_Value_Test: {Max_Value_Test}")
    print(f"Min_Value_Test: {Min_Value_Test}")

    field_info_train_tensors = {}
    field_info_test_tensors = {}
    # Load the dataset of all fields & Randomly pick-out sensor points for the selected Field_idx
    with open(f'data_split/data_split_MRPT_Features_{Case_Num}_{field_idx}_{N_selected}.pic', 'rb') as fp: 
        data_split = pickle.load(fp)

        U_train          = data_split.U_train.to(device)
        Y_train          = data_split.Y_train.to(device)
        G_train          = data_split.G_train.to(device)

        Gin_train  		 = data_split.Gin_train.to(device)  		
        Yin_train   	 = data_split.Yin_train.to(device)

        LR_T_train  	 = data_split.LR_T_train.to(device)
        field_info_train_tensors[f'field_info_T'] = LR_T_train

        LR_P_train  	 = data_split.LR_P_train.to(device)
        field_info_train_tensors[f'field_info_P'] = LR_P_train

        LR_Vx_train  	 = data_split.LR_Vx_train.to(device)
        field_info_train_tensors[f'field_info_Vx'] = LR_Vx_train

        LR_Vy_train  	 = data_split.LR_Vy_train.to(device)
        field_info_train_tensors[f'field_info_Vy'] = LR_Vy_train

        LR_O2_train  	 = data_split.LR_O2_train.to(device)  	
        field_info_train_tensors[f'field_info_O2'] = LR_O2_train

        LR_CO2_train  	 = data_split.LR_CO2_train.to(device)  	
        field_info_train_tensors[f'field_info_CO2'] = LR_CO2_train

        LR_H2O_train  	 = data_split.LR_H2O_train.to(device)  	
        field_info_train_tensors[f'field_info_H2O'] = LR_H2O_train

        LR_CO_train		 = data_split.LR_CO_train.to(device)	
        field_info_train_tensors[f'field_info_CO'] = LR_CO_train

        LR_H2_train		 = data_split.LR_H2_train.to(device)	
        field_info_train_tensors[f'field_info_H2'] = LR_H2_train

        LR_Unified_train = data_split.LR_Unified_train.to(device)

        U_test           = data_split.U_test.to(device)
        Y_test           = data_split.Y_test.to(device)
        G_test           = data_split.G_test.to(device)

        Gin_test  		 = data_split.Gin_test.to(device)  		
        Yin_test   	 	 = data_split.Yin_test.to(device)  

        LR_T_test  	 	 = data_split.LR_T_test.to(device)
        field_info_test_tensors[f'field_info_T'] = LR_T_test

        LR_P_test  	 	 = data_split.LR_P_test.to(device)
        field_info_test_tensors[f'field_info_P'] = LR_P_test

        LR_Vx_test  	 = data_split.LR_Vx_test.to(device) 
        field_info_test_tensors[f'field_info_Vx'] = LR_Vx_test

        LR_Vy_test  	 = data_split.LR_Vy_test.to(device)  	
        field_info_test_tensors[f'field_info_Vy'] = LR_Vy_test

        LR_O2_test  	 = data_split.LR_O2_test.to(device)	
        field_info_test_tensors[f'field_info_O2'] = LR_O2_test

        LR_CO2_test  	 = data_split.LR_CO2_test.to(device)	
        field_info_test_tensors[f'field_info_CO2'] = LR_CO2_test

        LR_H2O_test  	 = data_split.LR_H2O_test.to(device)
        field_info_test_tensors[f'field_info_H2O'] = LR_H2O_test

        LR_CO_test		 = data_split.LR_CO_test.to(device)
        field_info_test_tensors[f'field_info_CO'] = LR_CO_test

        LR_H2_test		 = data_split.LR_H2_test.to(device)
        field_info_test_tensors[f'field_info_H2'] = LR_H2_test

        LR_Unified_test  = data_split.LR_Unified_test.to(device)

        n_inputF = U_train.shape[-1]
        n_pointD = Y_train.shape[-1]

    # Define the network
    if (NET_TYPE == 0):
        net = Mutual_SensorToFeature_InterInference_LoadFeature(layer_sizes, Final_layer_sizes, PreTrained_net).to(device)
        # Fix the parameters in the petrained net
        for param in net.PreTrained_net.parameters():
            param.requires_grad = False
    elif (NET_TYPE == 1): # Directly map the sensor values to the latent space
        layer_sizes = [N_selected] + hidden_sizes + [n_field_info]  
        net = Direct_SensorToFeature(layer_sizes).to(device)
    else:
        print('Net is not correctly defined !!!')
        exit()

    # Wrap the model with DataParallel
    net = nn.DataParallel(net, device_ids=device_ids)
    optimizer = optim.Adam(net.parameters(), lr=0.0002, weight_decay=1.0E-5) # , weight_decay=5.0E-5

    # Set up early stopping parameters
    patience = 100
    best_combined_loss = float('5.0') #Initial threshold to determine early stopping
    counter = 0
    train_loss_weight = 0.15
    test_loss_weight = 0.85
    criterion = nn.MSELoss(reduction='mean')

    field_weights = torch.tensor([1.0] * 9)  # Replace with actual weights if needed
    field_weights = field_weights.to(device)

    for epoch in range(N_EPOCH):

        train_loss   = 0
        train_losses = torch.zeros(len(field_names)+1, device=device)
        test_loss    = 0
        test_losses  = torch.zeros(len(field_names)+1, device=device)

        Total_Field_train_loss_Data = 0.0
        Total_Field_test_loss_Data  = 0.0
        Field_train_loss = torch.zeros(len(field_names), device=device)  # [len(field_names), len(field_names)]
        Field_test_loss  = torch.zeros(len(field_names), device=device)  # [len(field_names), len(field_names)]

        train_batch_count = 0
        test_batch_count  = 0

        start_time = time.time()  # Start time measurement

        for U, Y, G, Yin, Gin in get_data_iter(U_train, Y_train, G_train, Yin_train, Gin_train):
            loss = 0.0
            optimizer.zero_grad()

            #----------------------------
            # Entering the training phase:
            #----------------------------
            if (NET_TYPE == 0):
                Predicted_Features, Unified_Feature_output_train = net(Yin, Gin, num_heads, Min_Value_Train, Max_Value_Train)
            elif (NET_TYPE == 1):
                Unified_Feature_output_train = net(Yin, Gin,)
            else:
                print('Net is not correctly defined !!!')
                exit()

            for id in range( len(field_names) ):
                field_output = Predicted_Features[:, :, id]
                field_target = field_info_train_tensors[f'field_info_{field_names[id]}']
                field_loss = criterion(field_output, field_target)
                train_losses[id] = field_loss.item()
                
                train_loss += field_loss
                loss += field_weights[id] * field_loss

            mse_loss_Unifed_train = criterion(Unified_Feature_output_train, LR_Unified_train)

            train_losses[ len(field_names) ] = mse_loss_Unifed_train
            train_loss += mse_loss_Unifed_train
            loss += Unifed_weight *mse_loss_Unifed_train 

            loss.backward()
            optimizer.step()

            if EVALUATE is True and (epoch + 1) % N_REPORT == 0:
                with torch.no_grad():
                    # Evaluate the field-reconstruction performance in the training set
                    baseF = PreTrained_net.PosNet(Y)
                    DeNormalized_train_output = Denormalize_data(Unified_Feature_output_train, Min_Value_Train[len(field_names)], Max_Value_Train[len(field_names)]) # This is the recovered "Global_Unified_U"
                    # DeNormalized_train_output = Denormalize_data(LR_Unified_train, Min_Value_Train[len(field_names)], Max_Value_Train[len(field_names)]) # This is the recovered "Global_Unified_U"

                    Global_Unified_field_outputs = []
                    for field_name, field_net in PreTrained_net.field_nets.items(): # Generate all the fields
                        coef = field_net(DeNormalized_train_output)
                        combine = coef * baseF
                        Global_Unified_field_output = torch.sum(combine, dim=2, keepdim=True)
                        Global_Unified_field_outputs.append(Global_Unified_field_output)
                    Global_Unified_Gout = torch.cat(Global_Unified_field_outputs, dim=-1) #   All the field_idx(th) field results from Global_Unified_U

                    field_loss_data, field_losses = custom_mse_loss(Global_Unified_Gout, G, field_weights)
                    Total_Field_train_loss_Data += field_loss_data
                    for j, loss_item in enumerate(field_losses):
                        Field_train_loss[j] += loss_item

            train_batch_count += 1
        train_loss /= train_batch_count
        train_losses /= train_batch_count
        Total_Field_train_loss_Data /= train_batch_count
        Field_train_loss /= train_batch_count

        #----------------------------
        #   Entering the test phase:
        #----------------------------
        with torch.no_grad():  
            for U, Y, G, Yin, Gin in get_data_iter(U_test, Y_test, G_test, Yin_test, Gin_test):
                if (NET_TYPE == 0):
                    Predicted_Features, Unified_Feature_output_test = net(Yin, Gin, num_heads, Min_Value_Test, Max_Value_Test)
                elif (NET_TYPE == 1):
                    Unified_Feature_output_test = net(Yin, Gin)
                else:
                    print('Net is not correctly defined !!!')
                    exit()

                for id in range( len(field_names) ):
                    field_output = Predicted_Features[:, :, id]
                    field_target = field_info_test_tensors[f'field_info_{field_names[id]}']
                    field_loss = criterion(field_output, field_target)
                    test_losses[id] = field_loss.item()
                    
                    test_loss += field_loss
                
                mse_loss_Unifed_test = criterion(Unified_Feature_output_test, LR_Unified_test)
                test_losses[ len(field_names) ] = mse_loss_Unifed_test
                test_loss += mse_loss_Unifed_test

                if EVALUATE is True and (epoch + 1) % N_REPORT == 0:
                    baseF = PreTrained_net.PosNet(Y)
                    DeNormalized_test_output = Denormalize_data(Unified_Feature_output_test, Min_Value_Test[len(field_names)], Max_Value_Test[len(field_names)]) # This is the recovered "Global_Unified_U"
                    # DeNormalized_test_output = Denormalize_data(LR_Unified_test, Min_Value_Test[len(field_names)], Max_Value_Test[len(field_names)]) # This is the recovered "Global_Unified_U"

                    Global_Unified_field_outputs = []
                    for field_name, field_net in PreTrained_net.field_nets.items(): # Generate all the fields
                        coef = field_net(DeNormalized_test_output)
                        combine = coef * baseF
                        Global_Unified_field_output = torch.sum(combine, dim=2, keepdim=True)
                        Global_Unified_field_outputs.append(Global_Unified_field_output)
                    Global_Unified_Gout = torch.cat(Global_Unified_field_outputs, dim=-1) #   All the field_idx(th) field results from Global_Unified_U

                    field_loss_data, field_losses = custom_mse_loss(Global_Unified_Gout, G, field_weights)
                    Total_Field_test_loss_Data += field_loss_data
                    for j, loss_item in enumerate(field_losses):
                        Field_test_loss[j] += loss_item

                test_batch_count += 1
        test_loss /= test_batch_count
        test_losses /= test_batch_count
        Total_Field_test_loss_Data /= test_batch_count
        Field_test_loss /= test_batch_count

        if EVALUATE is False:
            combined_loss = train_loss_weight*mse_loss_Unifed_train + test_loss_weight*mse_loss_Unifed_test
        elif EVALUATE is True:
            combined_loss = train_loss_weight*Total_Field_train_loss_Data + test_loss_weight*Total_Field_test_loss_Data
        else:
            print(f'EVALUATE is {EVALUATE}, not correctly defined!!!')
            exit(0)

        end_time = time.time()  # End time measurement
        epoch_duration = end_time - start_time  # Calculate duration

        # Print and write to CSV 
        if (epoch + 1) % N_REPORT == 0:

            print(f'Epoch {epoch+1}/{N_EPOCH}, Duration: {epoch_duration:.4f} seconds')
            print()

            print(f'Epoch {epoch+1}/{N_EPOCH}, for feature retrieving, Train Loss: {train_loss.item()}, Test Loss: {test_loss.item()}')
            for id in range( len(field_names) ):
                print(f'Train Loss for field {field_names[id]}: {train_losses[id].item()}, Test Loss for field {field_names[id]}: {test_losses[id].item()}')
            print(f'Train Loss for Unifed feature: {train_losses[ len(field_names) ].item()}, Test Loss for Unifed feature: {test_losses[ len(field_names) ].item()}')
            print()

            if EVALUATE is True:
                print(f'Total Field Train Loss: {Total_Field_train_loss_Data.item()}, Total Field Test Loss: {Total_Field_test_loss_Data.item()}')
                for id, field_name in enumerate(field_names):
                    print(f'Train Loss for field {field_name}: {Field_train_loss[id].item()}, Test Loss for field {field_name}: {Field_test_loss[id].item()}')
                print()

                with open(f'Loss_csv/train_test_loss_reconstruction/Reconstruct_loss_{NET_NAME[NET_TYPE]}.csv', 'at', newline='') as fp:
                    writer = csv.writer(fp, delimiter='\t')
                    if ((epoch + 1) // N_REPORT == 1):
                        header = ['Epoch', 'Overall_Train_Loss', 'Overall_Test_Loss']
                        interleaved_field_names = [fn for field_name in field_names for fn in (f'Train_{field_name}_loss', f'Test_{field_name}_loss')]
                        header.extend(interleaved_field_names)
                        writer.writerow(header)

                    row_data = [epoch + 1, Total_Field_train_loss_Data.item(), Total_Field_test_loss_Data.item()] 
                    # Combine train and test losses for each field into pairs
                    for field_train_loss, field_test_loss in zip(Field_train_loss, Field_test_loss):
                        row_data.append(f'{field_train_loss.item()}')
                        row_data.append(f'{field_test_loss.item()}')
                    writer.writerow(row_data)

            # Write to CSV file
            with open(f'Loss_csv/train_test_loss_reconstruction/train_test_loss_S2F_{NET_NAME[NET_TYPE]}.csv', 'at', newline='') as fp:
                writer = csv.writer(fp, delimiter='\t')
                if ((epoch + 1) // N_REPORT == 1):
                    fp.write(NET_SETTINGS)
                    header = ['Epoch', 'Train_Loss', 'Test_Loss']
                    interleaved_field_names = ['Train_Unifed_loss', 'Test_Unified_loss']
                    header.extend(interleaved_field_names)
                    interleaved_field_names = [fn for name in field_names + ['Unified_feature'] for fn in (f'Train_{name}_loss', f'Test_{name}_loss')]
                    header.extend(interleaved_field_names)
                    writer.writerow(header)
                
                row_data = [epoch + 1, train_loss.item(), test_loss.item()] + [train_losses[ len(field_names) ].item(), test_losses[ len(field_names) ].item()] + \
                            [item for pair in zip(train_losses.tolist(), test_losses.tolist()) for item in pair]
                writer.writerow(row_data)

            # Determine whether should trigger early stopping
            if ( combined_loss < best_combined_loss ):
                best_combined_loss = combined_loss
                print(f'Best combined loss so far is {best_combined_loss}, still improving')
                counter = 0
                
                model_save_path = 'Output_Net/net_{}_state_dict.pth'.format(NET_NAME[NET_TYPE])
                torch.save(net.module.state_dict(), model_save_path)
                print('Successfully saved the latest best net at {}'.format(model_save_path))
                print('...Successfully Saved net.pic')
                print()

                # Export the results to a new CSV file & evaluate the field-reconstruction performance
                with torch.no_grad():
                    with open(f'LatentRepresentation/predictions_{NET_NAME[NET_TYPE]}.csv', 'w', newline='') as file:
                        csv_writer = csv.writer(file)
                        csv_writer.writerow(['u_data', 'Global_Unified_U', 'Predicted_Global_Unified_U'])
                        csv_writer.writerow(['Training:'])
                        for i in range(U.shape[0]):
                            row = [U[i].cpu().numpy().tolist(),
                                LR_Unified_train[i].cpu().numpy().tolist(),
                                Unified_Feature_output_train[i].cpu().numpy().tolist()]
                            csv_writer.writerow(row)

                        csv_writer.writerow(['Testing:'])
                        for i in range(U.shape[0]):
                            row = [U[i].cpu().numpy().tolist(),
                                LR_Unified_test[i].cpu().numpy().tolist(),
                                Unified_Feature_output_test[i].cpu().numpy().tolist()]
                            csv_writer.writerow(row)

            else:
                counter += 1
                print(f'Best combined loss so far is {best_combined_loss}, NOT further improving')
                print(f'The counter for triggering early stopping is {counter}')
                print()
                if counter >= patience:
                    print("Early stopping triggered")
                    break

