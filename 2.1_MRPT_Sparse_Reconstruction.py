
#__________________________________________________________________________________________________________________________________________
# In this version, the code will load the pretrained net
# Downstream tasks; Generate random sparse sensor points from the Data_split, recover the unified latent feature using a MLP, and then rebuild all the fields
#__________________________________________________________________________________________________________________________________________

import sys
sys.path.append('..')

import io
import csv
import numpy as np 

import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from torch import nn 
from constant import DataSplit 
from network import Direct_SensorToFeature, Mutual_Representation_PreTrain_Net, Mutual_SensorToFeature_InterInference

# Specify the GPUs to use
device_ids = [0]
device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

#__________________________PARAMETERS_________________________________
field_names = ['T', 'P', 'Vx', 'Vy', 'O2', 'CO2', 'H2O', 'CO', 'H2']
n_fields = len(field_names)
N_EPOCH = 1000000
Case_Num = 300

EVALUATE = True
INDICE_RENEW = True
ADD_NOISE = False
mu = 0
sigma = 0.03

n_field_info = 36
n_baseF = 40 
n_cond = 11 #Length of the condition vector in U

field_idx = 0 # The field used for sparse reconstruction, 4 = O2
N_selected = 25  # Points to be extracted for Y_select
N_P_Selected = 1000 # points to evaluate performances
#Transformer layer parameters
num_heads = 9
num_layers = 1
hidden_sizes = [300, 300]

min_val = -3.0
max_val = 3.0
Unifed_weight = 5.0

NET_TYPE = int(0) 
                # 0 = [Mutual_Representation_PreTrain_Net]; 1 = [Direct_SensorToFeature]
NET_SETTINGS = f'LR = 5E-4, weight_decay = 5.0E-5\tSelecting {N_selected} random sensors from {field_names[field_idx]} to recover the latent features\tADD_NOISE is {ADD_NOISE}\thidden_sizes = {hidden_sizes}\n'
NET_NAME = [f'MRPT_SensorToFeature_N{N_selected}', f'Direct_SensorToFeature_N{N_selected}']

PreTrained_Net_Name = 'net_MRPT_Standard_state_dict'
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

def get_data_iter(U, Y, G, Yin, Gin, batch_size = 360): # random sampling in each epoch
    num_examples = len(U)
    num_points = Y.shape[1]
    indices = list(range(num_examples))
    np.random.shuffle(indices)  # 样本的读取顺序是随机的
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
    normalized_data = (data - min_val) / (max_val - min_val)

    sqrt_normalized_data = torch.sqrt(torch.clamp(normalized_data, min=0))
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
        # Export the results to a new CSV file & evaluate the field-reconstruction performance
        with open(f'LatentRepresentation/predictions_Train_{NET_NAME[NET_TYPE]}.csv', 'wt') as fp:
            pass
        with open(f'LatentRepresentation/predictions_Test_{NET_NAME[NET_TYPE]}.csv', 'wt') as fp:
            pass

    # Load the pretrained net
    PreTrained_net = Mutual_Representation_PreTrain_Net(n_field_info, n_baseF, num_heads, num_layers, num_fields=len(field_names)).to(device)
    state_dict = torch.load(Load_file_path)
    PreTrained_net.load_state_dict(state_dict)

    # Load the dataset of all fields & create random point selection & Add noise to the selected sensors
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

        indices_path = f'indices/Y_select_indices_C{Case_Num}_N{N_selected}.pkl'
        if INDICE_RENEW is True:
            Y_select_indices = torch.randperm(Y_train.size(1))[:N_selected].numpy()

            with open(indices_path, 'wb') as f:
                pickle.dump(Y_select_indices, f)
            print(f'Successfully saved Y_select_indices at {indices_path}')
        else:
            with open(indices_path, 'rb') as f:
                Y_select_indices = pickle.load(f)
            print(f'Successfully loaded Y_select_indices at {indices_path}')
        Yin_train = Y_train[:, Y_select_indices, :].to(device)
        Yin_test = Y_test[:, Y_select_indices, :].to(device)
        print('Y_train.shape = ', Y_train.shape)
        print('Yin_train.shape = ', Yin_train.shape)

        # Extract the temperature values (field_idx = 0) from G_train and G_test or other SELECTED field
        Gin_train = G_train[:, Y_select_indices, field_idx].unsqueeze(-1).to(device)  # Select the first field (temperature)
        Gin_test = G_test[:, Y_select_indices, field_idx].unsqueeze(-1).to(device)  # Select the first field (temperature)
        print('Gin_Train.shape = ', Gin_train.shape)

        if ADD_NOISE is True:
            # Generate Gaussian noise
            noise_train = torch.randn_like(Gin_train) * sigma + mu
            noise_test = torch.randn_like(Gin_test) * sigma + mu
            # Add the noise to the original tensors
            Gin_train = Gin_train + noise_train
            Gin_test  = Gin_test + noise_test

        # Save Y_select & the corresponding temperature value to a CSV file
        with open(f'Y_select/Y_select_Train_{NET_NAME[NET_TYPE]}.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Case', 'Point Index', 'X Coordinate', 'Y Coordinate', 'Temperature'])
            for case in range(Yin_train.size(0)):
                for i, point_idx in enumerate(Y_select_indices):  # Removed .cpu().numpy()
                    csvwriter.writerow([case, point_idx, Yin_train[case, i, 0].item(), Yin_train[case, i, 1].item(), Gin_train[case, i, 0].item()])

        # Save Y_select & the corresponding temperature value to a CSV file
        with open(f'Y_select/Y_select_Test_{NET_NAME[NET_TYPE]}.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Case', 'Point Index', 'X Coordinate', 'Y Coordinate', 'Temperature'])
            for case in range(Yin_train.size(0)):
                for i, point_idx in enumerate(Y_select_indices):  # Removed .cpu().numpy()
                    csvwriter.writerow([case, point_idx, Yin_train[case, i, 0].item(), Yin_train[case, i, 1].item(), Gin_train[case, i, 0].item()])

    # Define the network
    if (NET_TYPE == 0):
        layer_sizes = [n_field_info * 9] + hidden_sizes + [n_field_info]  
        net = Mutual_SensorToFeature_InterInference(layer_sizes, PreTrained_net).to(device)

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
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5.0E-5) 

    # Set up early stopping parameters
    patience = 100
    best_combined_loss = float('5.0') #Initial threshold to determine early stopping
    counter = 0
    train_loss_weight = 0.15
    test_loss_weight = 0.85
    criterion = nn.MSELoss(reduction='mean')

    field_weights = torch.tensor([1.0] * 9)  # Replace with actual weights if needed
    # field_weights = torch.tensor([5.0,0.2,2.0,2.0,5.0,1.0,1.0,1.0,1.0])
    field_weights = field_weights.to(device)

    for epoch in range(N_EPOCH):

        train_loss = 0
        train_losses = torch.zeros(len(field_names)+1, device=device)
        test_loss = 0
        test_losses = torch.zeros(len(field_names)+1, device=device)
        
        Total_Field_train_loss_Data = 0.0
        Total_Field_test_loss_Data = 0.0
        Field_train_loss = torch.zeros(len(field_names), device=device)  # [len(field_names), len(field_names)]
        Field_test_loss = torch.zeros(len(field_names), device=device)  # [len(field_names), len(field_names)]

        train_batch_count = 0
        test_batch_count = 0

        for U, Y, G, Yin, Gin in get_data_iter(U_train, Y_train, G_train, Yin_train, Gin_train):
            
            loss = 0
            optimizer.zero_grad()
            # print('Out there, Yin.shape is', Yin.shape)

            #   (1) Obtain the predicted features using the randomly selected sparse measurements ______________________________________________________________________________________________________________
            #       Extract latent features from the pre-trained net and normalize them, inclduing Train_ALL_Unified_U and Train_Global_Unified_U ______________________________________________________________
            #       Calculate the losses between the features from pre-trained net and sparsely recovered features _____________________________________________________________________________________________            
            if (NET_TYPE == 0):
                Predicted_Features, Unified_Feature_output = net(Yin, Gin, num_heads, min_val, max_val)
            elif (NET_TYPE == 1):
                Unified_Feature_output = net(Yin, Gin)
            else:
                print('Net is not correctly defined !!!')
                exit()

            with torch.no_grad():
                baseF = PreTrained_net.PosNet(Y)   #   [n_batch, np_selected, n_dim -> n_base]
                U_Unified_list_train = []
                for id in range(n_fields):   # Reconstructing the id(th) field
                    field_info = PreTrained_net._compress_data(baseF, G, id, num_heads) # [n_batch, n_field_info, n_fields]: The latent representations from one Encoder towards the id(th) field
                    
                    U_Unified = PreTrained_net.FieldMerges[id](field_info, id) #   [n_batch, n_field_info]: Unified latent representations from one Encoder towards the id(th) field
                    # print('U_Unified.shape is', U_Unified.shape)
                    U_Unified_list_train.append(U_Unified)

                ALL_Unified_U = torch.stack(U_Unified_list_train, dim=2) #   [n_batch, n_field_info, n_fields]
                Global_Unified_U = PreTrained_net.FinalMerge(ALL_Unified_U, -1)   #   [n_batch, n_field_info]
                # print('Global_Unified_U.shape is', Global_Unified_U.shape)

            Train_ALL_Unified_U = normalize_data(ALL_Unified_U, min_val, max_val)
            Train_Global_Unified_U = normalize_data(Global_Unified_U, min_val, max_val)

            if (NET_TYPE == 0): # This is only calculated when the Encoder is reused in fine-tuning stage
                for id in range(n_fields):
                    field_output = Predicted_Features[:, :, id]
                    field_target = Train_ALL_Unified_U[:, :, id]
                    field_loss = criterion(field_output, field_target)
                    train_losses[id] += field_loss.item()
                    
                    train_loss += field_loss
                    loss += field_weights[id] * field_loss

            train_loss_Unifed = criterion(Unified_Feature_output, Train_Global_Unified_U)
            train_losses[ len(field_names) ] += train_loss_Unifed
            train_loss += train_loss_Unifed
            loss += Unifed_weight * train_loss_Unifed

            loss.backward()
            optimizer.step()

            #   (2) Upon evaluation, denormalize the Unified_Feature_output, and reconstruct all the fields using the pre-trained net _____________________________________________________________________________
            #       Calculate the field reconstruction losses in the train set ____________________________________________________________________________________________________________________________________
            if EVALUATE is True and (epoch + 1) % 10 == 0:
                with torch.no_grad():

                    # Export the results to a new CSV file & evaluate the field-reconstruction performance
                    with open('LatentRepresentation/predictions_Train_{}.csv'.format(NET_NAME[NET_TYPE]), 'w', newline='') as file:
                        csv_writer = csv.writer(file)
                        csv_writer.writerow(['u_data', 'Global_Unified_U', 'Predicted_Global_Unified_U'])
                        csv_writer.writerow(['Training:'])

                        for i in range(U.shape[0]):
                            row = [U[i].cpu().numpy().tolist(),
                                Train_Global_Unified_U[i].cpu().numpy().tolist(),
                                Unified_Feature_output[i].cpu().numpy().tolist()]
                            csv_writer.writerow(row)

                    # Evaluate the field-reconstruction performance in the training set
                    DeNormalized_train_output = Denormalize_data(Unified_Feature_output, min_val, max_val) # This is the recovered "Global_Unified_U"

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

        #   (3) After the training loop, enter the test loop and further finish the evaluation steps _______________________________________________________________________________________________________
        #       Now, load the data in the test set, and then obtain the predicted features using the randomly selected sparse measurements _________________________________________________________________
        #       Calculate the losses between the features from pre-trained net and sparsely recovered features _____________________________________________________________________________________________
        if (epoch + 1) % 10 == 0:

            with torch.no_grad():  # Use no_grad for evaluation in test phase
                #   for U, Y, G, ID in get_data_iter(U_test, Yin_test, Gin_test, test_info_data):
                for U, Y, G, Yin, Gin in get_data_iter(U_test, Y_test, G_test, Yin_test, Gin_test):

                    if (NET_TYPE == 0):
                        Predicted_Features, Unified_Feature_output = net(Yin, Gin, num_heads, min_val, max_val)
                    elif (NET_TYPE == 1):
                        Unified_Feature_output = net(Yin, Gin)
                    else:
                        print('Net is not correctly defined !!!')
                        exit()

                    #   Extract latent features from the pre-trained net
                    baseF = PreTrained_net.PosNet(Y)   #   [n_batch, np_selected, n_dim -> n_base]
                    U_Unified_list_test = []
                    for id in range(n_fields):   # Reconstructing the field_idx(th) field
                        field_info = PreTrained_net._compress_data(baseF, G, id, num_heads) # [n_batch, n_field_info, n_fields]: The latent representations from one Encoder towards the field_idx(th) field
                        # print('field_info.shape is', field_info.shape)
                        
                        U_Unified = PreTrained_net.FieldMerges[id](field_info, id) #   [n_batch, n_field_info]: Unified latent representations from one Encoder towards the field_idx(th) field
                        # print('U_Unified.shape is', U_Unified.shape)
                        U_Unified_list_test.append(U_Unified)

                    ALL_Unified_U = torch.stack(U_Unified_list_test, dim=2) #   [n_batch, n_field_info, n_fields]
                    Global_Unified_U = PreTrained_net.FinalMerge(ALL_Unified_U, -1)   #   [n_batch, n_field_info]
                    # print('Global_Unified_U.shape is', Global_Unified_U.shape)

                    Test_ALL_Unified_U = normalize_data(ALL_Unified_U, min_val, max_val)
                    Test_Global_Unified_U = normalize_data(Global_Unified_U, min_val, max_val)

                    if (NET_TYPE == 0):
                        for id in range(n_fields):
                            field_output = Predicted_Features[:, :, id]
                            # print('field_output.shape is ', field_output.shape)
                            field_target = Test_ALL_Unified_U[:, :, id]
                            # print('field_target.shape is ', field_target.shape)
                            field_loss = criterion(field_output, field_target)
                            test_losses[id] += field_loss.item()                        
                            test_loss += field_loss

                    test_loss_Unifed = criterion(Unified_Feature_output, Test_Global_Unified_U)
                    test_losses[ len(field_names) ] += test_loss_Unifed
                    test_loss += test_loss_Unifed

            #   (4) Upon evaluation, Write out the pre-trained features and sparsely recovered features _____________________________________________________________________________________________________
            #       Also, denormalize the Unified_Feature_output, and reconstruct all the fields using the pre-trained net in the test set __________________________________________________________________
            #        Calculate the field reconstruction losses in the test set ______________________________________________________________________________________________________________________________
                    if EVALUATE is True:
                        # Export the results to a new CSV file & evaluate the field-reconstruction performance
                        with open('LatentRepresentation/predictions_Test_{}.csv'.format(NET_NAME[NET_TYPE]), 'w', newline='') as file:
                            csv_writer = csv.writer(file)
                            csv_writer.writerow(['Testing:'])

                            for i in range(U.shape[0]):
                                row = [U[i].cpu().numpy().tolist(),
                                    Test_Global_Unified_U[i].cpu().numpy().tolist(),
                                    Unified_Feature_output[i].cpu().numpy().tolist()]
                                csv_writer.writerow(row)

                        # Evaluate the field-reconstruction performance in the test set
                        DeNormalized_test_output = Denormalize_data(Unified_Feature_output, min_val, max_val) # This is the recovered "Global_Unified_U"
                        with torch.no_grad():
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

            #   (5) Finished all the works, now print and write the rest contents ________________________________________________________________________________________________________________

                # Write to CSV file
                with open(f'Loss_csv/train_test_loss_reconstruction/Reconstruct_loss_{NET_NAME[NET_TYPE]}.csv', 'at', newline='') as fp:
                    writer = csv.writer(fp, delimiter='\t')
                    if ((epoch + 1) // 10 == 1):
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

            # combined_loss = train_loss_weight*Total_train_loss_Data + test_loss_weight*Total_test_loss_Data
            # combined_loss = train_loss_weight*train_loss + test_loss_weight*test_loss
            # combined_loss = train_loss_weight*train_losses[n_fields] + test_loss_weight*test_losses[n_fields]
            combined_loss = train_loss_weight*Total_Field_train_loss_Data + test_loss_weight*Total_Field_test_loss_Data

            print(f'Epoch {epoch+1}/{N_EPOCH}, Train Loss: {train_loss.item()}, Test Loss: {test_loss.item()}')
            for id in range( len(field_names) ):
                print(f'Train Loss for field {field_names[id]}: {train_losses[id].item()}, Test Loss for field {field_names[id]}: {test_losses[id].item()}')
            print(f'Train Loss for Unifed feature: {train_losses[ len(field_names) ].item()}, Test Loss for Unifed feature: {test_losses[ len(field_names) ].item()}')
            print()

            if EVALUATE is True:
                print(f'Total Field Train Loss: {Total_Field_train_loss_Data.item()}, Total Field Test Loss: {Total_Field_test_loss_Data.item()}')
                for id, field_name in enumerate(field_names):
                    print(f'Train Loss for field {field_name}: {Field_train_loss[id].item()}, Test Loss for field {field_name}: {Field_test_loss[id].item()}')
                print()

            # Write S2F loss to CSV file
            with open(f'Loss_csv/train_test_loss_reconstruction/train_test_loss_S2F_{NET_NAME[NET_TYPE]}.csv', 'at', newline='') as fp:
                writer = csv.writer(fp, delimiter='\t')
                if ((epoch + 1) // 10 == 1):
                    fp.write(NET_SETTINGS)
                    header = ['Epoch', 'Train_Loss', 'Test_Loss']
                    # interleaved_field_names = ['Train_Unifed_loss', 'Test_Unified_loss']
                    # header.extend(interleaved_field_names)
                    interleaved_field_names = [fn for name in field_names + ['Unified_feature'] for fn in (f'Train_{name}_loss', f'Test_{name}_loss')]
                    header.extend(interleaved_field_names)
                    writer.writerow(header)
                
                # row_data = [epoch + 1, train_loss.item(), test_loss.item()] + [train_losses[ len(field_names) ].item(), test_losses[ len(field_names) ].item()] + \
                row_data = [epoch + 1, train_loss.item(), test_loss.item()] + [item for pair in zip(train_losses.tolist(), test_losses.tolist()) for item in pair]
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

            else:
                counter += 1
                print(f'Best combined loss so far is {best_combined_loss}, NOT further improving')
                print(f'The counter for triggering early stopping is {counter}')
                print()
                if counter >= patience:
                    print("Early stopping triggered")
                    break

