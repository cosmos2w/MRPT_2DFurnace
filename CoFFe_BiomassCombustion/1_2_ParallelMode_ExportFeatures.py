import os
import csv
import torch
import pickle
from constant import DataSplit_F, DataSplit_STD
from network import CoFFe_PreTrain_Net_ParallelMode

# Specify the GPUs to use
device_ids = [0]
device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

#__________________________PARAMETERS_________________________________

# Set the neccessary parameters based on the corresponding pre-training task

n_field_info = 36
n_baseF = 50 
num_heads = 6
num_layers = 1

field_idx   = 0     # The selected field for sparse reconstruction
N_selected  = 25    # The number of random sensor points to be selected for sparse reconstruction
field_names = ['T', 'P', 'Vx', 'Vy', 'O2', 'CO2', 'H2O', 'CO', 'H2']

n_fields = len(field_names)
N_P_Evaluation = 1000
EVALUATION = True

PreTrained_Net_Name = 'net_CoFFe_ParallelMode_state_dict'
Load_file_path = f'Output_Net/{PreTrained_Net_Name}.pth'
outFile = f'data_split/CoFFe_Parallel_Features_From{field_idx}_{N_selected}.pic'
#____________________________________________________________________

def get_data_iter(U, Y, G, Yin, Gin, batch_size = 360): 
    num_examples = len(U)
    num_points = Y.shape[1]
    indices = list(range(num_examples))
    # np.random.shuffle(indices)  
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) 
        j = j.to(device)

        selected_points = torch.randperm(num_points)[:N_P_Evaluation].to(device)
        yield  U.index_select(0, j), Y.index_select(0, j).index_select(1, selected_points), G.index_select(0, j).index_select(1, selected_points), Yin.index_select(0, j), Gin.index_select(0, j)

def field_mse_loss(output, target, field_weights):

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

def standardize_features(latent_vectors):
    mean = torch.mean(latent_vectors, dim=0, keepdim=True)
    std = torch.std(latent_vectors, dim=0, keepdim=True)
    
    epsilon = 1e-8
    std = torch.clamp(std, min=epsilon)
    
    standardized_vectors = (latent_vectors - mean) / std
    
    return standardized_vectors, mean, std

if __name__ == '__main__':

    for id in range(len(field_names) + 1):
        field_name_or_feature = 'Unified' if id == len(field_names) else field_names[id]
        with open(f'LatentRepresentation/ParallelMode/FeaturesFrom_{PreTrained_Net_Name}_To_{field_name_or_feature}.csv', 'wt') as fp: 
            pass

    # Load the pretrained net
    PreTrained_net = CoFFe_PreTrain_Net_ParallelMode(n_field_info, n_baseF, num_heads, num_layers, num_fields=len(field_names)).to(device)
    state_dict = torch.load(Load_file_path)
    PreTrained_net.load_state_dict(state_dict)

    field_weights = torch.tensor([1.0] * 9)  # Replace with modified weights if needed
    field_weights = field_weights.to(device)

    # Load the dataset of all fields
    with open('data_split/data_split.pic', 'rb') as fp: 
        data_split = pickle.load(fp)

        U_train = data_split.U_train.to(device)
        Y_train = data_split.Y_train.to(device)
        G_train = data_split.G_train.to(device)

        U_test = data_split.U_test.to(device)
        Y_test = data_split.Y_test.to(device)
        G_test = data_split.G_test.to(device)

        Y_select_indices = torch.randperm(Y_train.size(1))[:N_selected].numpy()
        Yin_train = Y_train[:, Y_select_indices, :].to(device)
        Yin_test = Y_test[:, Y_select_indices, :].to(device)
        print('Y_train.shape = ', Y_train.shape)
        print('Yin_train.shape = ', Yin_train.shape)

        # Extract the selected field values (field_idx) from G_train and G_test
        Gin_train = G_train[:, Y_select_indices, field_idx].unsqueeze(-1).to(device)  
        Gin_test = G_test[:, Y_select_indices, field_idx].unsqueeze(-1).to(device)  
        print('Gin_Train.shape = ', Gin_train.shape)
    
    with torch.no_grad():
        for U, Y, G, Yin, Gin in get_data_iter(U_train, Y_train, G_train, Yin_train, Gin_train):

            baseF = PreTrained_net.PosNet(Y)   #   [n_batch, np_selected, n_dim -> n_base]
            FieldInfo_train = PreTrained_net._compress_data(baseF, G, num_heads)
            print('In train set, FieldInfo_train.shape is ', FieldInfo_train.shape)
            Unified_FieldInfo_train = PreTrained_net.FinalMerge(FieldInfo_train, -1)   #   [n_batch, n_field_info]
            print('In train set, Unified_FieldInfo_train.shape is ', Unified_FieldInfo_train.shape)

            if EVALUATION is True:
                Total_Field_train_loss_Data = 0.0
                Field_train_loss = torch.zeros(len(field_names), device=device)  # [len(field_names), len(field_names)]

                Unified_field_outputs = []   # This means, the field output is decoded from the "Unified" feature
                for field_name, field_net in PreTrained_net.field_nets.items(): # Generate all the fields
                    coef = field_net(Unified_FieldInfo_train)
                    combine = coef * baseF
                    Unified_field_output = torch.sum(combine, dim=2, keepdim=True)
                    Unified_field_outputs.append(Unified_field_output)
                Unified_Gout_train = torch.cat(Unified_field_outputs, dim=-1) #   All the field_idx(th) field results from Unified_U

                field_loss_data, field_losses = field_mse_loss(Unified_Gout_train, G, field_weights)
                Total_Field_train_loss_Data = field_loss_data
                for j, loss_item in enumerate(field_losses):
                    Field_train_loss[j] = loss_item                

                print(f'Total Field Train Loss: {Total_Field_train_loss_Data.item()}')
                for id, field_name in enumerate(field_names):
                    print(f'Train Loss for field {field_name}: {Field_train_loss[id].item()}')
                print()

        for U, Y, G, Yin, Gin in get_data_iter(U_test, Y_test, G_test, Yin_test, Gin_test):

            baseF = PreTrained_net.PosNet(Y)   #   [n_batch, np_selected, n_dim -> n_base]
            FieldInfo_test = PreTrained_net._compress_data(baseF, G, num_heads)
            print('In test set, FieldInfo_test.shape is ', FieldInfo_test.shape)
            Unified_FieldInfo_test = PreTrained_net.FinalMerge(FieldInfo_test, -1)   #   [n_batch, n_field_info]
            print('In test set, Unified_FieldInfo_test.shape is ', Unified_FieldInfo_test.shape)

            if EVALUATION is True:
                Total_Field_test_loss_Data = 0.0
                Field_test_loss = torch.zeros(len(field_names), device=device)  

                Unified_field_outputs = [] 
                for field_name, field_net in PreTrained_net.field_nets.items(): 
                    coef = field_net(Unified_FieldInfo_test)
                    combine = coef * baseF
                    Unified_field_output = torch.sum(combine, dim=2, keepdim=True)
                    Unified_field_outputs.append(Unified_field_output)
                Unified_Gout_test = torch.cat(Unified_field_outputs, dim=-1) 

                field_loss_data, field_losses = field_mse_loss(Unified_Gout_test, G, field_weights)
                Total_Field_test_loss_Data = field_loss_data
                for j, loss_item in enumerate(field_losses):
                    Field_test_loss[j] = loss_item                

                print(f'Total Field Test Loss: {Total_Field_test_loss_Data.item()}')
                for id, field_name in enumerate(field_names):
                    print(f'Test Loss for field {field_name}: {Field_test_loss[id].item()}')
                print()

    std_field_info_train_tensors = {}
    std_field_info_test_tensors  = {}

    train_mean_tensors = {}
    train_std_tensors  = {}
    test_mean_tensors  = {}
    test_std_tensors   = {}

    # Perform Normalization to latent features and Write to CSV file
    for id in range(len(field_names) + 1):
        # Determine field name or unified feature
        field_name_or_feature = 'Unified' if id == len(field_names) else field_names[id]
        
        info_train = Unified_FieldInfo_train if id == len(field_names) else FieldInfo_train[:, :, id]
        info_test  = Unified_FieldInfo_test  if id == len(field_names) else FieldInfo_test[:, :, id]

        #   Perform standardization for the latent representations
        info_train_std, mean_train, std_train = standardize_features(info_train)
        info_test_std, mean_test, std_test    = standardize_features(info_test)

        if id == len(field_names): # For the global feature
            std_U_Unified_train = info_train_std
            std_U_Unified_test  = info_test_std

            Unified_mean_train = mean_train 
            Unified_std_train  = std_train 
            Unified_mean_test  = mean_test 
            Unified_std_test   = std_test 
        else:
            std_field_info_train_tensors[f'field_info_{field_names[id]}'] = info_train_std
            std_field_info_test_tensors[f'field_info_{field_names[id]}']  = info_test_std

            train_mean_tensors[f'field_info_{field_names[id]}'] = mean_train
            train_std_tensors[f'field_info_{field_names[id]}']  = std_train
            test_mean_tensors[f'field_info_{field_names[id]}']  = mean_test
            test_std_tensors[f'field_info_{field_names[id]}']   = std_test

        print(f'Successfully Standardized FeaturesFrom_{PreTrained_Net_Name}_To_{field_name_or_feature}!')

        with open(f'LatentRepresentation/ParallelMode/FeaturesFrom_{PreTrained_Net_Name}_To_{field_name_or_feature}.csv', 'wt') as fp:
            simple_writer = csv.writer(fp)

            fieldnames = ['U', 'field_info']
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            print(U_train.shape)
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
    data_split = DataSplit_STD(
            U_train          = U_train,
            Y_train          = Y_train,
            G_train          = G_train,
            Gin_train  		 = Gin_train,
            Yin_train   	 = Yin_train,

            LR_T_train  	 = std_field_info_train_tensors['field_info_T'],
            MEAN_T_train     = train_mean_tensors['field_info_T'],
            STD_T_train  	 = train_std_tensors['field_info_T'],

            LR_P_train  	 = std_field_info_train_tensors['field_info_P'],
            MEAN_P_train     = train_mean_tensors['field_info_P'],
            STD_P_train  	 = train_std_tensors['field_info_P'],

            LR_Vx_train      = std_field_info_train_tensors['field_info_Vx'],
            MEAN_Vx_train    = train_mean_tensors['field_info_Vx'],
            STD_Vx_train     = train_std_tensors['field_info_Vx'],

            LR_Vy_train      = std_field_info_train_tensors['field_info_Vy'],
            MEAN_Vy_train    = train_mean_tensors['field_info_Vy'],
            STD_Vy_train     = train_std_tensors['field_info_Vy'],

            LR_O2_train  	 = std_field_info_train_tensors['field_info_O2'],
            MEAN_O2_train    = train_mean_tensors['field_info_O2'],
            STD_O2_train  	 = train_std_tensors['field_info_O2'],

            LR_CO2_train  	 = std_field_info_train_tensors['field_info_CO2'],
            MEAN_CO2_train   = train_mean_tensors['field_info_CO2'],
            STD_CO2_train  	 = train_std_tensors['field_info_CO2'],

            LR_H2O_train     = std_field_info_train_tensors['field_info_H2O'],
            MEAN_H2O_train   = train_mean_tensors['field_info_H2O'],
            STD_H2O_train    = train_std_tensors['field_info_H2O'],

            LR_CO_train      = std_field_info_train_tensors['field_info_CO'],
            MEAN_CO_train    = train_mean_tensors['field_info_CO'],
            STD_CO_train     = train_std_tensors['field_info_CO'],

            LR_H2_train      = std_field_info_train_tensors['field_info_H2'],
            MEAN_H2_train    = train_mean_tensors['field_info_H2'],
            STD_H2_train     = train_std_tensors['field_info_H2'],

            LR_Unified_train = std_U_Unified_train,
            MEAN_Unified_train = Unified_mean_train,
            STD_Unified_train  = Unified_std_train,

            U_test           = U_test,
            Y_test           = Y_test,
            G_test           = G_test,
            Gin_test  		 = Gin_test,
            Yin_test   	 	 = Yin_test,

            LR_T_test  	     = std_field_info_test_tensors['field_info_T'],
            MEAN_T_test      = test_mean_tensors['field_info_T'],
            STD_T_test  	 = test_std_tensors['field_info_T'],

            LR_P_test  	     = std_field_info_test_tensors['field_info_P'],
            MEAN_P_test      = test_mean_tensors['field_info_P'],
            STD_P_test  	 = test_std_tensors['field_info_P'],

            LR_Vx_test       = std_field_info_test_tensors['field_info_Vx'],
            MEAN_Vx_test     = test_mean_tensors['field_info_Vx'],
            STD_Vx_test      = test_std_tensors['field_info_Vx'],
 
            LR_Vy_test       = std_field_info_test_tensors['field_info_Vy'],
            MEAN_Vy_test     = test_mean_tensors['field_info_Vy'],
            STD_Vy_test      = test_std_tensors['field_info_Vy'],

            LR_O2_test  	 = std_field_info_test_tensors['field_info_O2'],
            MEAN_O2_test     = test_mean_tensors['field_info_O2'],
            STD_O2_test  	 = test_std_tensors['field_info_O2'],

            LR_CO2_test  	 = std_field_info_test_tensors['field_info_CO2'],
            MEAN_CO2_test    = test_mean_tensors['field_info_CO2'],
            STD_CO2_test  	 = test_std_tensors['field_info_CO2'],

            LR_H2O_test      = std_field_info_test_tensors['field_info_H2O'],
            MEAN_H2O_test    = test_mean_tensors['field_info_H2O'],
            STD_H2O_test     = test_std_tensors['field_info_H2O'],
 
            LR_CO_test       = std_field_info_test_tensors['field_info_CO'],
            MEAN_CO_test     = test_mean_tensors['field_info_CO'],
            STD_CO_test      = test_std_tensors['field_info_CO'],
 
            LR_H2_test       = std_field_info_test_tensors['field_info_H2'],
            MEAN_H2_test     = test_mean_tensors['field_info_H2'],
            STD_H2_test      = test_std_tensors['field_info_H2'],

            LR_Unified_test  = std_U_Unified_test,
            MEAN_Unified_test = Unified_mean_test,
            STD_Unified_test  = Unified_std_test
        )

    with open(outFile, 'wb') as fp:
        pickle.dump(data_split, fp)
        print(f'...Data saved to Data_split {outFile} !!!')

