
import sys
sys.path.append('..')

import csv
import torch
import pickle
from constant import DataSplit_F
from network import Mutual_Representation_PreTrain_Net

# Specify the GPUs to use
device_ids = [1]
device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

# Set the constants based on pre-training task
Case_Num = 300
n_field_info = 36
n_baseF = 40 

field_names = ['T', 'P', 'Vx', 'Vy', 'O2', 'CO2', 'H2O', 'CO', 'H2']
field_idx = 0 # The selected field for sparse reconstruction
n_fields = len(field_names)
N_selected = 15 #   Number of sensor points selected for field reconstruction
N_P_Selected = 1200
EVALUATION = True

#Transformer layer parameters
num_heads = 6
num_layers = 1

PreTrained_Net_Name = 'net_MRPT_Standard_200_state_dict'
Load_file_path = 'Output_Net/{}.pth'.format(PreTrained_Net_Name)

outFile = f'data_split/data_split_MRPT_Features_{Case_Num}_{field_idx}_{N_selected}.pic'

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

if __name__ == '__main__':

    for id in range(len(field_names) + 1):
        field_name_or_feature = 'Unified' if id == len(field_names) else field_names[id]
        with open(f'LatentRepresentation/FeaturesFrom_{PreTrained_Net_Name}_To_{field_name_or_feature}.csv', 'wt') as fp: 
            pass

    # Load the pretrained net
    PreTrained_net = Mutual_Representation_PreTrain_Net(n_field_info, n_baseF, num_heads, num_layers, num_fields=len(field_names)).to(device)
    state_dict = torch.load(Load_file_path)
    PreTrained_net.load_state_dict(state_dict)

    field_weights = torch.tensor([1.0] * 9)  # Replace with actual weights if needed
    # field_weights = torch.tensor([5.0,0.2,2.0,2.0,5.0,1.0,1.0,1.0,1.0])
    field_weights = field_weights.to(device)

    # Load the dataset of all fields
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

        Y_select_indices = torch.randperm(Y_train.size(1))[:N_selected].numpy()
        Yin_train = Y_train[:, Y_select_indices, :].to(device)
        Yin_test = Y_test[:, Y_select_indices, :].to(device)
        print('Y_train.shape = ', Y_train.shape)
        print('Yin_train.shape = ', Yin_train.shape)

        # Extract the temperature values (field_idx = 0) from G_train and G_test
        Gin_train = G_train[:, Y_select_indices, field_idx].unsqueeze(-1).to(device)  
        Gin_test = G_test[:, Y_select_indices, field_idx].unsqueeze(-1).to(device)  
        print('Gin_Train.shape = ', Gin_train.shape)
    
    with torch.no_grad():
        for U, Y, G, Yin, Gin in get_data_iter(U_train, Y_train, G_train, Yin_train, Gin_train):

            baseF = PreTrained_net.PosNet(Y)   #   [n_batch, np_selected, n_dim -> n_base]
            U_Unified_list_train = []
            for id in range(n_fields):   # Reconstructing the id(th) field
                field_info = PreTrained_net._compress_data(baseF, G, id, num_heads) # [n_batch, n_field_info, n_fields]: The latent representations from one Encoder towards the id(th) field
                
                U_Unified = PreTrained_net.FieldMerges[id](field_info, id) #   [n_batch, n_field_info]: Unified latent representations from one Encoder towards the id(th) field
                U_Unified_list_train.append(U_Unified)

            ALL_Unified_U_train = torch.stack(U_Unified_list_train, dim=2) #   [n_batch, n_field_info, n_fields]
            print('In train set, ALL_Unified_U_train.shape is ', ALL_Unified_U_train.shape)
            Global_Unified_U_train = PreTrained_net.FinalMerge(ALL_Unified_U_train, -1)   #   [n_batch, n_field_info]
            print('In train set, Global_Unified_U_train.shape is ', Global_Unified_U_train.shape)

            if EVALUATION is True:
                Total_Field_train_loss_Data = 0.0
                Field_train_loss = torch.zeros(len(field_names), device=device)  # [len(field_names), len(field_names)]

                Global_Unified_field_outputs = []   # This means, the field output is decoded from the "Global_Unified" feature
                for field_name, field_net in PreTrained_net.field_nets.items(): # Generate all the fields
                    coef = field_net(Global_Unified_U_train)
                    combine = coef * baseF
                    Global_Unified_field_output = torch.sum(combine, dim=2, keepdim=True)
                    Global_Unified_field_outputs.append(Global_Unified_field_output)
                Global_Unified_Gout = torch.cat(Global_Unified_field_outputs, dim=-1) #   All the field_idx(th) field results from Global_Unified_U

                field_loss_data, field_losses = custom_mse_loss(Global_Unified_Gout, G, field_weights)
                Total_Field_train_loss_Data = field_loss_data
                for j, loss_item in enumerate(field_losses):
                    Field_train_loss[j] = loss_item                

                print(f'Total Field Train Loss: {Total_Field_train_loss_Data.item()}')
                for id, field_name in enumerate(field_names):
                    print(f'Train Loss for field {field_name}: {Field_train_loss[id].item()}')
                print()

        for U, Y, G, Yin, Gin in get_data_iter(U_test, Y_test, G_test, Yin_test, Gin_test):

            baseF = PreTrained_net.PosNet(Y)   #   [n_batch, np_selected, n_dim -> n_base]
            U_Unified_list_test = []
            for id in range(n_fields):   # Reconstructing the id(th) field
                field_info = PreTrained_net._compress_data(baseF, G, id, num_heads) # [n_batch, n_field_info, n_fields]: The latent representations from one Encoder towards the id(th) field
                
                U_Unified = PreTrained_net.FieldMerges[id](field_info, id) #   [n_batch, n_field_info]: Unified latent representations from one Encoder towards the id(th) field
                U_Unified_list_test.append(U_Unified)

            ALL_Unified_U_test = torch.stack(U_Unified_list_test, dim=2) #   [n_batch, n_field_info, n_fields]
            print('In test set, ALL_Unified_U_test.shape is ', ALL_Unified_U_test.shape)
            Global_Unified_U_test = PreTrained_net.FinalMerge(ALL_Unified_U_test, -1)   #   [n_batch, n_field_info]
            print('In test set, Global_Unified_U_test.shape is ', Global_Unified_U_test.shape)

            if EVALUATION is True:
                Total_Field_test_loss_Data = 0.0
                Field_test_loss = torch.zeros(len(field_names), device=device)  # [len(field_names), len(field_names)]

                Global_Unified_field_outputs = [] # This means, the field output is decoded from the "Global_Unified" feature
                for field_name, field_net in PreTrained_net.field_nets.items(): # Generate all the fields
                    coef = field_net(Global_Unified_U_test)
                    combine = coef * baseF
                    Global_Unified_field_output = torch.sum(combine, dim=2, keepdim=True)
                    Global_Unified_field_outputs.append(Global_Unified_field_output)
                Global_Unified_Gout = torch.cat(Global_Unified_field_outputs, dim=-1) #   All the field_idx(th) field results from Global_Unified_U

                field_loss_data, field_losses = custom_mse_loss(Global_Unified_Gout, G, field_weights)
                Total_Field_test_loss_Data = field_loss_data
                for j, loss_item in enumerate(field_losses):
                    Field_test_loss[j] = loss_item                

                print(f'Total Field Test Loss: {Total_Field_test_loss_Data.item()}')
                for id, field_name in enumerate(field_names):
                    print(f'Test Loss for field {field_name}: {Field_test_loss[id].item()}')
                print()

    normalized_field_info_train_tensors = {}
    normalized_field_info_test_tensors  = {}

    normalized_U_Unified_train = torch.empty_like(Global_Unified_U_train)
    normalized_U_Unified_test  = torch.empty_like(Global_Unified_U_test)

    # Perform Normalization to latent features and Write to CSV file
    for id in range(len(field_names) + 1):
        # Determine field name or unified feature
        field_name_or_feature = 'Unified' if id == len(field_names) else field_names[id]
        
        info_train = Global_Unified_U_train if id == len(field_names) else ALL_Unified_U_train[:, :, id]
        info_test  = Global_Unified_U_test  if id == len(field_names) else ALL_Unified_U_test[:, :, id]

        max_value_train = torch.max(info_train)
        min_value_train = torch.min(info_train)
        max_value_test = torch.max(info_test)
        min_value_test = torch.min(info_test)

        #   Perform normalization for the latent representations
        normalized_info_train = (info_train - min_value_train) / (max_value_train - min_value_train)
        normalized_info_test = (info_test - min_value_test) / (max_value_test - min_value_test)
        # Apply sqrt, clamping to avoid taking sqrt of negative numbers
        normalized_info_train = torch.sqrt(torch.clamp(normalized_info_train, min=0))
        normalized_info_test = torch.sqrt(torch.clamp(normalized_info_test, min=0))

        if id == len(field_names): # For the global feature
            normalized_U_Unified_train = normalized_info_train
            normalized_U_Unified_test = normalized_info_test
        else:
            normalized_field_info_train_tensors[f'field_info_{field_names[id]}'] = normalized_info_train
            normalized_field_info_test_tensors[f'field_info_{field_names[id]}'] = normalized_info_test

        print(f'Successfully Normalized FeaturesFrom_{PreTrained_Net_Name}_To_{field_name_or_feature}!')

        with open(f'LatentRepresentation/FeaturesFrom_{PreTrained_Net_Name}_To_{field_name_or_feature}.csv', 'wt') as fp:

            # Standard csv.writer for writing the initial scalars row
            simple_writer = csv.writer(fp)
            # Write a single row for max/min values
            simple_writer.writerow([
                'Max Value Train', 'Min Value Train', 'Max Value Test', 'Min Value Test'
            ])
            simple_writer.writerow([
                max_value_train.item(), min_value_train.item(), max_value_test.item(), min_value_test.item()
            ])

            fieldnames = ['U', 'field_info']
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()

            # Write data from the training set
            writer.writerow({'U': 'From training set:'})
            for i in range(U_train.shape[0]):
                row = {'U': U_train[i].cpu().numpy().tolist(),
                    'field_info': info_train[i].cpu().numpy().tolist()}
                writer.writerow(row)            
            # Write data from the test set
            writer.writerow({'U': 'From test set:'})
            for i in range(U_test.shape[0]):
                row = {'U': U_test[i].cpu().numpy().tolist(),
                    'field_info': info_test[i].cpu().numpy().tolist()}
                writer.writerow(row)
        print(f'Successfully export FeaturesFrom_{PreTrained_Net_Name}_To_{field_name_or_feature}.csv!\n')

    #   Export all the data to DataSplit for next-step training
    data_split = DataSplit_F(
            U_train          = U_train,
            Y_train          = Y_train,
            G_train          = G_train,
            Gin_train  		 = Gin_train,
            Yin_train   	 = Yin_train,
            LR_T_train  	 = normalized_field_info_train_tensors['field_info_T'],
            LR_P_train  	 = normalized_field_info_train_tensors['field_info_P'],
            LR_Vx_train  	 = normalized_field_info_train_tensors['field_info_Vx'],
            LR_Vy_train  	 = normalized_field_info_train_tensors['field_info_Vy'],
            LR_O2_train  	 = normalized_field_info_train_tensors['field_info_O2'],
            LR_CO2_train  	 = normalized_field_info_train_tensors['field_info_CO2'],
            LR_H2O_train  	 = normalized_field_info_train_tensors['field_info_H2O'],
            LR_CO_train		 = normalized_field_info_train_tensors['field_info_CO'],
            LR_H2_train		 = normalized_field_info_train_tensors['field_info_H2'],
            LR_Unified_train = normalized_U_Unified_train,

            U_test           = U_test,
            Y_test           = Y_test,
            G_test           = G_test,
            Gin_test  		 = Gin_test,
            Yin_test   	 	 = Yin_test,
            LR_T_test  	 	 = normalized_field_info_test_tensors['field_info_T'],
            LR_P_test  	 	 = normalized_field_info_test_tensors['field_info_P'],
            LR_Vx_test  	 = normalized_field_info_test_tensors['field_info_Vx'],
            LR_Vy_test  	 = normalized_field_info_test_tensors['field_info_Vy'],
            LR_O2_test  	 = normalized_field_info_test_tensors['field_info_O2'],
            LR_CO2_test  	 = normalized_field_info_test_tensors['field_info_CO2'],
            LR_H2O_test  	 = normalized_field_info_test_tensors['field_info_H2O'],
            LR_CO_test		 = normalized_field_info_test_tensors['field_info_CO'],
            LR_H2_test		 = normalized_field_info_test_tensors['field_info_H2'],
            LR_Unified_test  = normalized_U_Unified_test
        )

    with open(outFile, 'wb') as fp:
        pickle.dump(data_split, fp)
        print(f'...Data saved to Data_split {outFile} !!!')

