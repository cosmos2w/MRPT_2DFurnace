
import pickle
import torch 
import numpy as np 

from torch import nn 
from constant import DataSplit 
from torch.nn import functional as F

device_ids = [0]
device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

class BranchNet(nn.Module):
    def __init__(self, layer_size, activation=True):
        super(BranchNet, self).__init__()
        self.activation = activation
        self.layers = nn.ModuleList() 
        for i in range(len(layer_size) - 1):
            self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1], bias=True))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = torch.tanh(x)

        if self.activation:
            x = torch.sigmoid(x)
        x = x.unsqueeze(1)
        return x

class TrunkNet(nn.Module):
    def __init__(self, layer_size, activation=True):
        super(TrunkNet, self).__init__()
        self.activation = activation
        self.layers = nn.ModuleList() 
        for i in range(len(layer_size) - 1):
            self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1], bias=True))

    def forward(self, x):
        n_batch = x.shape[0]
        n_point = x.shape[1]
        n_dim = x.shape[2]

        x = x.reshape(-1, n_dim)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = torch.tanh(x)

        if self.activation:
            x = torch.sigmoid(x)
        x = x.reshape(n_batch, n_point, -1)
        return x

class MLP(nn.Module):
    def __init__(self, layer_size):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList() 
        for i in range(len(layer_size) - 1):
            self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1], bias=True))
        
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x

class MultiField_AttentionLayer(nn.Module):
    def __init__(self, n_field_info, num_heads, num_fields, dropout_rate = 0.10):
        super().__init__()
        self.num_fields = num_fields

        self.Qnet_MH_list = nn.ModuleList([nn.Linear(n_field_info, n_field_info * num_heads) for _ in range(num_fields)])
        self.Knet_MH_list = nn.ModuleList([nn.Linear(n_field_info, n_field_info * num_heads) for _ in range(num_fields)])
        self.Vnet_MH_list = nn.ModuleList([nn.Linear(n_field_info, n_field_info * num_heads) for _ in range(num_fields)])
        self.output_linear_list = nn.ModuleList([nn.Linear(n_field_info * num_heads, n_field_info) for _ in range(num_fields)])

        self.dropout = nn.Dropout(dropout_rate)
        self.norms = nn.ModuleList([nn.LayerNorm(n_field_info) for _ in range(num_fields)])
        self.mlps = nn.ModuleList([MLP(layer_size=[n_field_info, 2 * n_field_info, n_field_info]) for _ in range(num_fields)])

    def forward(self, field_info, field_idx, num_heads):
        batch_size, seq_length, _ = field_info.size()

        Q = self.Qnet_MH_list[field_idx](field_info)
        K = self.Knet_MH_list[field_idx](field_info)
        V = self.Vnet_MH_list[field_idx](field_info)

        # Split the matrices for multi-heads and reshape for batched matrix multiplication
        _, _, depth = Q.size()
        Q = Q.view(batch_size, seq_length, num_heads, depth // num_heads).transpose(1, 2)
        K = K.view(batch_size, seq_length, num_heads, depth // num_heads).transpose(1, 2)
        V = V.view(batch_size, seq_length, num_heads, depth // num_heads).transpose(1, 2)

        # Scaled dot-product attention for each head
        QK = torch.matmul(Q, K.transpose(-2, -1))
        attention_scores = F.softmax(QK / ((depth // num_heads) ** 0.5), dim=-1)
        weighted_values = torch.matmul(attention_scores, V)

        # Concatenate heads and reshape
        weighted_values = weighted_values.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        weighted_values = self.dropout(weighted_values)  # Apply dropout to the attention output
        
        # Mix the information from all heads and reduce dimensionality
        output = self.output_linear_list[field_idx](weighted_values)

        mlp_output = self.mlps[field_idx](output)
        
        residual = field_info + mlp_output  # Add the residual connection
        output = self.norms[field_idx](residual)

        return output

class AttentionMergeLayer(nn.Module): 
    def __init__(self, n_field_info, n_fields, dropout_rate=0.1):
        super(AttentionMergeLayer, self).__init__()
        self.n_field_info = n_field_info
        self.n_fields = n_fields

        self.query = nn.Linear(n_fields, n_field_info)
        self.key =   nn.Linear(n_fields, n_field_info)
        self.value = nn.Linear(n_fields, n_field_info)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.mlp = MLP(layer_size=[n_field_info, 2 * n_field_info, n_field_info])
        self.norm = nn.LayerNorm(n_field_info)
        self.norm2 = nn.LayerNorm(n_field_info)

    def forward(self, U, field_idx):

        Q = self.query(U)
        K = self.key(U)
        V = self.value(U)

        # Calculate attention scores
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.n_fields ** 0.5)
        attention_weights = self.softmax(attention_scores)
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights
        unified_feature_vector = torch.bmm(attention_weights, V)
        output = unified_feature_vector.mean(dim=2)

        # Residual connection
        residual = U.mean(dim=2)  # Reducing along the n_field dimension to match output shape

        if field_idx > 0:
            Self_feature = U[:, :, field_idx] # extract the latent feature to reconstruct the field_idx(th) field extracted from itself
            output += Self_feature
        else:
            output += residual

        return output

class CoFFe_PreTrain_Net_ParallelMode(nn.Module): 
    def __init__(self, n_field_info, n_base, num_heads, num_layers, num_fields):
        print("num_heads", num_heads)
        print("num_fields", num_fields)
        print("n_base", n_base)
        super().__init__()

        self.MapperNet = nn.ModuleList([TrunkNet(layer_size=[ n_base + 1 , 60, 60, n_field_info]) for _ in range(num_fields)])

        # Create multiple layers of the transformer
        self.Field_AttentionLayers = nn.ModuleList([
            MultiField_AttentionLayer(n_field_info, num_heads, num_fields, dropout_rate = 0.30) for _ in range(num_layers)
        ])
        
        self.FinalMerge = AttentionMergeLayer(n_field_info, num_fields, dropout_rate = 0.05) 

        self.PosNet = TrunkNet(layer_size=[2, 50, 50, 50, n_base])
        self.field_nets = nn.ModuleDict({
            'T':   BranchNet([n_field_info, 50, 50, n_base]),
            'P':   BranchNet([n_field_info, 50, 50, n_base]),
            'Vx':  BranchNet([n_field_info, 50, 50, n_base]),
            'Vy':  BranchNet([n_field_info, 50, 50, n_base]),
            'O2':  BranchNet([n_field_info, 50, 50, n_base]),
            'CO2': BranchNet([n_field_info, 50, 50, n_base]),
            'H2O': BranchNet([n_field_info, 50, 50, n_base]),
            'CO':  BranchNet([n_field_info, 50, 50, n_base]),
            'H2':  BranchNet([n_field_info, 50, 50, n_base])
        })

    def _compress_data(self, Y, Gin, num_heads):
        n_fields = Gin.shape[-1]
        compressed_info_list = []
        for field_idx in range(n_fields):
            
            Gin_Y =  torch.cat((Y, Gin[:, :, field_idx:field_idx+1]), dim=2) # [n_batch, n_points, n_encoding + 1] * n_fields            
            base_for_Gin_Y = self.MapperNet[field_idx](Gin_Y)
            
            field_info = base_for_Gin_Y         
            for layer in self.Field_AttentionLayers:
                field_info = layer(field_info, field_idx, num_heads)

            compressed_info = field_info.mean(dim=1)
            compressed_info_list.append(compressed_info)
        
        compressed_info_ALL = torch.stack(compressed_info_list, dim=-1)
        return compressed_info_ALL

    def forward(self, cond, Y, Gin, num_heads):

        baseF = self.PosNet(Y)

        field_info = self._compress_data(baseF, Gin, num_heads)        
        U = field_info
        U_Unified = self.FinalMerge(U, -1)
        
        # Compute outputs for each field using the corresponding network
        field_outputs = {}
        for field_idx, (field_name, field_net) in enumerate(self.field_nets.items()):
            
            U_sliced = U[:, :, field_idx]
            coef = field_net(U_sliced)
            combine = coef * baseF
            field_output = torch.sum(combine, dim=2, keepdim=True)
            field_outputs[field_name] = field_output

        # Compute outputs for each field using the corresponding network
        field_outputs_Unified = {}
        for field_name, field_net in self.field_nets.items(): # Generate all the fields
            
            coef = field_net(U_Unified)
            combine = coef * baseF
            field_output = torch.sum(combine, dim=2, keepdim=True)
            field_outputs_Unified[field_name] = field_output        

        Gout_list = []
        Gout = torch.cat(list(field_outputs.values()), dim=-1)
        Gout = Gout.view(U.size(0), -1, len(self.field_nets))  
        Gout_list.append(Gout)

        Gout_Unified = torch.cat(list(field_outputs_Unified.values()), dim=-1)
        Gout_Unified = Gout_Unified.view(U.size(0), -1, len(self.field_nets))  
        Gout_list.append(Gout_Unified)

        return Gout_list

class CoFFe_PreTrain_Net_MutualDecodingMode(nn.Module): 
    def __init__(self, n_field_info, n_base, num_heads, num_layers, num_fields=9):
        print("num_heads is", num_heads)
        print("num_fields is", num_fields)
        print("n_field_info is", n_field_info)
        super().__init__()

        self.MapperNet = nn.ModuleList([TrunkNet(layer_size=[ n_base + 1 , 50, 50, n_field_info]) for _ in range(num_fields)])

        self.Field_AttentionLayers = nn.ModuleList([
            MultiField_AttentionLayer(n_field_info, num_heads, num_fields, dropout_rate = 0.40) for _ in range(num_layers)
        ])
        
        self.FinalMerge = AttentionMergeLayer(n_field_info, num_fields, dropout_rate = 0.10)  

        self.MLPs = nn.ModuleList([MLP(layer_size=[n_field_info, 60, n_field_info]) for _ in range(num_fields)])
        self.PosNet = TrunkNet(layer_size=[2, 60, 60, 60, n_base])
        self.field_nets = nn.ModuleDict({
            'T':   BranchNet([n_field_info, 50, 50, n_base]),
            'P':   BranchNet([n_field_info, 50, 50, n_base]),
            'Vx':  BranchNet([n_field_info, 50, 50, n_base]),
            'Vy':  BranchNet([n_field_info, 50, 50, n_base]),
            'O2':  BranchNet([n_field_info, 50, 50, n_base]),
            'CO2': BranchNet([n_field_info, 50, 50, n_base]),
            'H2O': BranchNet([n_field_info, 50, 50, n_base]),
            'CO':  BranchNet([n_field_info, 50, 50, n_base]),
            'H2':  BranchNet([n_field_info, 50, 50, n_base])
        })

    def _compress_data(self, Y, Gin, num_heads): 
        n_fields = Gin.shape[-1]

        compressed_info_list = []
        for field_idx in range(n_fields):
            
            Gin_Y =  torch.cat((Y, Gin[:, :, field_idx:field_idx+1]), dim=2) # [n_batch, n_points, n_encoding + 1] * n_fields
            
            base_for_Gin_Y = self.MapperNet[field_idx](Gin_Y)
            field_info = base_for_Gin_Y
            
            for layer in self.Field_AttentionLayers:
                field_info = layer(field_info, field_idx, num_heads)

            compressed_info = field_info.mean(dim=1)
            compressed_info_list.append(compressed_info)
                   
        compressed_info_ALL = torch.stack(compressed_info_list, dim=-1)
        return compressed_info_ALL

    def forward(self, cond, Y, Gin, num_heads):
        n_batch = cond.shape[0]
        n_fields = Gin.shape[-1]
        baseF = self.PosNet(Y)

        field_info = self._compress_data(baseF, Gin, num_heads)

        U = field_info
        U_Unified = self.FinalMerge(U, -1)

        Gout_list = []
        for field_idx in range(n_fields + 1): # Includes an extra iteration for the unified feature
            # Determine the input tensor: U_sliced for fields or U_Unified for unified feature
            U_input = U[:, :, field_idx] if field_idx < n_fields else U_Unified
            
            # Compute outputs for each field using the corresponding network
            field_outputs = []
            for field_name, field_net in self.field_nets.items(): # Generate all the fields
                coef = field_net(U_input)
                combine = coef * baseF
                field_output = torch.sum(combine, dim=2, keepdim=True)
                field_outputs.append(field_output)
            
            # Stack all field outputs and reshape
            Gout = torch.cat(field_outputs, dim=-1)
            Gout = Gout.view(U.size(0), -1, len(self.field_nets))  # Assuming the second dimension is correctly sized
            Gout_list.append(Gout)

        return Gout_list

class CoFFe_PreTrain_Net_MutualEncodingMode(nn.Module): 
    def __init__(self, n_field_info, n_base, num_heads, num_layers, num_fields):
        print("num_heads is", num_heads)
        print("num_fields is", num_fields)
        print("n_field_info is", n_field_info)
        super().__init__()

        self.MapperNet = nn.ModuleList([TrunkNet(layer_size=[ n_base + 1 , 50, 50, n_field_info ]) for _ in range(num_fields)])

        # Create multiple layers of MH transformers for encoders towards different fields
        self.Field_AttentionLayers = nn.ModuleList([
            MultiField_AttentionLayer(n_field_info, num_heads, num_fields, dropout_rate = 0.50) for _ in range(num_layers)
        ])
        
        self.FieldMerges = nn.ModuleList([
            AttentionMergeLayer(n_field_info, num_fields, dropout_rate = 0.10) for _ in range(num_fields)
        ])
        self.FinalMerge = AttentionMergeLayer(n_field_info, num_fields, dropout_rate = 0.10) 

        # Channels to process the field_info for different fields, including temperature field
        self.PosNet = TrunkNet(layer_size=[2, 50, 50, 50, n_base])
        self.field_names = ['T', 'P', 'Vx', 'Vy', 'O2', 'CO2', 'H2O', 'CO', 'H2']
        self.field_nets = nn.ModuleDict({
            'T':   BranchNet([n_field_info, 50, 50, n_base]),
            'P':   BranchNet([n_field_info, 50, 50, n_base]),
            'Vx':  BranchNet([n_field_info, 50, 50, n_base]),
            'Vy':  BranchNet([n_field_info, 50, 50, n_base]),
            'O2':  BranchNet([n_field_info, 50, 50, n_base]),
            'CO2': BranchNet([n_field_info, 50, 50, n_base]),
            'H2O': BranchNet([n_field_info, 50, 50, n_base]),
            'CO':  BranchNet([n_field_info, 50, 50, n_base]),
            'H2':  BranchNet([n_field_info, 50, 50, n_base])
        })

    def _compress_data(self, Y, Gin, field_idx, num_heads): # Y = baseF here
        n_fields = Gin.shape[-1]
        compressed_info_list = []
        for id in range(n_fields):  #   For all fields in Gin, pass them through the field_idx(th) Encoder, and obtain LRs towards the field_idx(th) field
            
            Gin_Y =  torch.cat((Y, Gin[:, :, id:id+1]), dim=2) # [n_batch, n_points, n_encoding + 1] * n_fields

            # The field_idx(th) Encoder:             
            field_info = self.MapperNet[field_idx](Gin_Y)            
            for layer in self.Field_AttentionLayers:
                field_info = layer(field_info, field_idx, num_heads)

            compressed_info = field_info.mean(dim=1)
            compressed_info_list.append(compressed_info)
                   
        compressed_info_ALL = torch.stack(compressed_info_list, dim=-1)
        return compressed_info_ALL

    def forward(self, cond, Y, Gin, num_heads):

        n_batch = cond.shape[0]
        n_fields = Gin.shape[-1]
        baseF = self.PosNet(Y)

        Gout_list = []
        U_Unified_list = []
        for field_idx in range(n_fields):   # Reconstructing the field_idx(th) field
            field_name = self.field_names[field_idx]
            
            # [n_batch, n_field_info, n_fields]: The latent representations from one Encoder towards the field_idx(th) field
            field_info = self._compress_data(baseF, Gin, field_idx, num_heads) 

            # [n_batch, n_field_info]: Unified latent representations from one Encoder towards the field_idx(th) field
            U_Unified = self.FieldMerges[field_idx](field_info, field_idx) 

            field_outputs = []
            for id in range(n_fields +1):
                U_input = field_info[:, :, id] if id < n_fields else U_Unified
                
                coef = self.field_nets[field_name](U_input)  #   Branch net for the field_idx(th) field
                combine = coef * baseF
                field_output = torch.sum(combine, dim=2, keepdim=True)  #   Reconstructed field_idx(th) field from the latent representation of the id(th) field or Unified
                field_outputs.append(field_output)
            
            # Stack all field outputs and reshape 
            Gout = torch.cat(field_outputs, dim=-1) #   All the field_idx(th) field results from different input fields, [n_batch, n_points, n_fields + 1]
            
            U_Unified_list.append(U_Unified)
            Gout_list.append(Gout)
        
        Global_Unified_U = torch.stack(U_Unified_list, dim=2) #   [n_batch, n_field_info, n_fields]
        Global_Unified_U = self.FinalMerge(Global_Unified_U, field_idx = -1)   #   [n_batch, n_field_info]

        Global_Unified_field_outputs = []
        for field_name, field_net in self.field_nets.items(): # Generate all the fields

            coef = field_net(Global_Unified_U)
            combine = coef * baseF

            Global_Unified_field_output = torch.sum(combine, dim=2, keepdim=True)
            Global_Unified_field_outputs.append(Global_Unified_field_output)
        Global_Unified_Gout = torch.cat(Global_Unified_field_outputs, dim=-1) #   All the field_idx(th) field results from Global_Unified_U

        return Gout_list, Global_Unified_Gout

# To predict UNSEEN variables based on pre-trained net
class UNSeen_FineTune_Net(nn.Module):
    def __init__(self, N_selected, n_field_info, n_base, PreTrained_net):
        super().__init__()
        
        if isinstance(PreTrained_net, nn.DataParallel):
            PreTrained_net = PreTrained_net.module
        
        # The Extra BranchNet in decoder for downstream unseen field to be trained
        self.field_net = BranchNet([n_field_info, 50, 50, n_base]) 

        # The TrunkNet will be used and trained if mode == False
        self.CondNet_Backup = BranchNet([N_selected, 50, 50, n_base])
        self.PosNet_Backup  = TrunkNet([2, 50, 50, 50, n_base])
        
        self.PreTrained_net = PreTrained_net
        self._compress_data = PreTrained_net._compress_data
        self.PosNet         = PreTrained_net.PosNet
        self.FieldMerges    = PreTrained_net.FieldMerges
        self.FinalMerge     = PreTrained_net.FinalMerge

    def forward(self, Unified_Feature_output, Y, Gin, num_heads, mode):

        if mode == True: # True means the reconstruction will proceed based on the pre-trained net 
            with torch.no_grad(): 
                baseF = self.PosNet(Y)

            coef = self.field_net(Unified_Feature_output)
        elif mode == False: # False means training a new DeepONet on the dataset of new variable
            baseF = self.PosNet(Y)
            Gin_flatten = Gin.squeeze(-1)
            coef = self.CondNet_Backup(Gin_flatten)

        combine = coef * baseF
        field_output = torch.sum(combine, dim=2, keepdim=True)

        return field_output

class Finetuning_SensorToFeatures_MutualEncoding(nn.Module):
    def __init__(self, layer_sizes, Final_layer_sizes, PreTrained_net, num_fields = 9):
        super(Finetuning_SensorToFeatures_MutualEncoding, self).__init__()
        self.n_fields = num_fields

        print('layer_sizes of MLP is ', layer_sizes)
        print('layer_sizes of Final MLP is ', Final_layer_sizes)
        if isinstance(PreTrained_net, nn.DataParallel):
            PreTrained_net = PreTrained_net.module

        self.MLPs = nn.ModuleList([MLP(layer_sizes) for _ in range(num_fields)]) 
        self.MLP_Final = MLP(Final_layer_sizes)
        self.field_names = ['T', 'P', 'Vx', 'Vy', 'O2', 'CO2', 'H2O', 'CO', 'H2']

        self.PreTrained_net = PreTrained_net
        self.MapperNet = PreTrained_net.MapperNet
        self.Field_AttentionLayers = PreTrained_net.Field_AttentionLayers
        self.FinalMerge = PreTrained_net.FinalMerge
        self.PosNet = PreTrained_net.PosNet

    def standardize_features(self, latent_vectors, mean, std):
        epsilon = 1e-8
        std = torch.clamp(std, min=epsilon)
        
        standardized_vectors = (latent_vectors - mean) / std
        return standardized_vectors

    def unstandardize_features(self, standardized_vectors, mean, std):
        return standardized_vectors * std + mean

    def forward(self, Yin, Gin, num_heads, mean_tensors, std_tensors):   # Yin and Gin are the sparse measurements of one field

        Base_Y = self.PosNet(Yin)   #   [n_batch, np_selected, n_dim -> n_base]
        Gin_Y =  torch.cat((Base_Y, Gin), dim=2)    #   [n_batch, np_selected, n_base + 1]

        #____________________________(1) PRE-TRAINED ENCODERs_____________________________________
        compressed_info_list = []
        STD_compressed_info_list = []
        for id in range(self.n_fields):  
            field_info = self.MapperNet[id](Gin_Y)      #   [n_batch, np_selected, n_field_info], the UP-LIFTING net in each encoder will "take a look"
            for layer in self.Field_AttentionLayers:    #   And then, the Transformer in each encoder will re-organize the information
                field_info = layer(field_info, id, num_heads)
            
            compressed_info = field_info.mean(dim=1)    #   [n_batch, n_field_info]
            STD_compressed_info = self.standardize_features(compressed_info, mean_tensors[f'field_info_{self.field_names[id]}'], std_tensors[f'field_info_{self.field_names[id]}'])
            
            compressed_info_list.append(compressed_info)
            STD_compressed_info_list.append(STD_compressed_info)
        compressed_info_ALL = torch.stack(compressed_info_list, dim=-1) # [n_batch, n_field_info, n_fields]: The latent representations from one field out of all Encoders
        STD_compressed_info_ALL = torch.stack(STD_compressed_info_list, dim=-1)

        #____________________________(2) Standardization and Correction______________________________
        compressed_info_concat = compressed_info_ALL.reshape(compressed_info_ALL.shape[0], -1)  # [n_batch, n_field_info * n_fields]
        STD_compressed_info_concat = STD_compressed_info_ALL.reshape(STD_compressed_info_ALL.shape[0], -1)  # [n_batch, n_field_info * n_fields]
        
        Predicted_Features_list = []
        for id in range(self.n_fields):
            Feature_id = self.MLPs[id](STD_compressed_info_concat) + STD_compressed_info_ALL[:, :, id]

            Predicted_Features_list.append(Feature_id)
        Predicted_Features = torch.stack(Predicted_Features_list, dim=-1)   # This is supposed to be the standardized features

        UnSTD_compressed_info_list = []
        for id in range(self.n_fields):  
            UnSTD_compressed_info = self.unstandardize_features(Predicted_Features[:, :, id], mean_tensors[f'field_info_{self.field_names[id]}'], std_tensors[f'field_info_{self.field_names[id]}'])
            UnSTD_compressed_info_list.append(UnSTD_compressed_info)
        UnSTD_compressed_info_ALL = torch.stack(UnSTD_compressed_info_list, dim=-1)

        #____________________________(3) Final merge and Correction________________________________
        Global_Unified_U = self.FinalMerge(UnSTD_compressed_info_ALL, -1)
        STD_Global_Unified_U = self.standardize_features(Global_Unified_U, mean_tensors[f'Unified'], std_tensors[f'Unified'])
        U_Unified = self.MLP_Final(STD_Global_Unified_U) + STD_Global_Unified_U  # Add a ResNet. The field_idx(th) mlp is used to map from the merged_unify to final output 

        return Predicted_Features, U_Unified

class Finetuning_SensorToFeatures_MutualDecoding(nn.Module):
    def __init__(self, layer_sizes, Final_layer_sizes, PreTrained_net, num_fields = 9):
        super().__init__()
        self.n_fields = num_fields
        self.field_names = ['T', 'P', 'Vx', 'Vy', 'O2', 'CO2', 'H2O', 'CO', 'H2']

        print('layer_sizes of MLP is ', layer_sizes)
        print('layer_sizes of Final MLP is ', Final_layer_sizes)
        if isinstance(PreTrained_net, nn.DataParallel):
            PreTrained_net = PreTrained_net.module

        self.MLPs = nn.ModuleList([MLP(layer_sizes) for _ in range(num_fields)]) 
        self.MLP_Final = MLP(Final_layer_sizes)

        self.PreTrained_net = PreTrained_net
        self.MapperNet = PreTrained_net.MapperNet
        self.Field_AttentionLayers = PreTrained_net.Field_AttentionLayers
        self.FinalMerge = PreTrained_net.FinalMerge
        self.PosNet = PreTrained_net.PosNet

    def standardize_features(self, latent_vectors, mean, std):
        epsilon = 1e-8
        std = torch.clamp(std, min=epsilon)
        
        standardized_vectors = (latent_vectors - mean) / std
        return standardized_vectors

    def unstandardize_features(self, standardized_vectors, mean, std):
        return standardized_vectors * std + mean

    def forward(self, Yin, Gin, num_heads, field_idx, mean_tensors, std_tensors):   # Yin and Gin are the sparse measurements of one field

        Base_Y = self.PosNet(Yin)   #   [n_batch, np_selected, n_dim -> n_base]
        Gin_Y =  torch.cat((Base_Y, Gin), dim=2)    #   [n_batch, np_selected, n_base + 1]

        #____________________________(1) PRE-TRAINED ENCODERs_____________________________________

        field_info = self.MapperNet[field_idx](Gin_Y)  #   [n_batch, np_selected, n_field_info], the UP-LIFTING net in each encoder will "take a look"
        for layer in self.Field_AttentionLayers:    #   And then, the Transformer in each encoder will re-organize the information
            field_info = layer(field_info, field_idx, num_heads)
        
        compressed_info = field_info.mean(dim=1)    #   [n_batch, n_field_info]
        STD_compressed_info = self.standardize_features(compressed_info, mean_tensors[f'field_info_{self.field_names[field_idx]}'], std_tensors[f'field_info_{self.field_names[field_idx]}'])

        #____________________________(2) Standardization and Correction______________________________
        
        Predicted_Features_list = []
        for id in range(self.n_fields):
            Feature_id = self.MLPs[id](STD_compressed_info) 
            Predicted_Features_list.append(Feature_id)
        Predicted_Features = torch.stack(Predicted_Features_list, dim=-1)   # This is supposed to be the standardized features

        UnSTD_compressed_info_list = []
        for id in range(self.n_fields):  
            UnSTD_compressed_info = self.unstandardize_features(Predicted_Features[:, :, id], mean_tensors[f'field_info_{self.field_names[id]}'], std_tensors[f'field_info_{self.field_names[id]}'])
            UnSTD_compressed_info_list.append(UnSTD_compressed_info)
        UnSTD_compressed_info_ALL = torch.stack(UnSTD_compressed_info_list, dim=-1)

        #____________________________(3) Final merge and Correction________________________________
        Global_Unified_U = self.FinalMerge(UnSTD_compressed_info_ALL, -1)
        STD_Global_Unified_U = self.standardize_features(Global_Unified_U, mean_tensors[f'Unified'], std_tensors[f'Unified'])
        U_Unified = self.MLP_Final(STD_Global_Unified_U) + STD_Global_Unified_U  # Add a ResNet. The field_idx(th) mlp is used to map from the merged_unify to final output 

        return Predicted_Features, U_Unified

class Finetuning_SensorToFeatures_ParallelMode(nn.Module):
    def __init__(self, layer_sizes, Final_layer_sizes, PreTrained_net, num_fields = 9):
        super().__init__()
        self.n_fields = num_fields
        self.field_names = ['T', 'P', 'Vx', 'Vy', 'O2', 'CO2', 'H2O', 'CO', 'H2']

        print('layer_sizes of MLP is ', layer_sizes)
        print('layer_sizes of Final MLP is ', Final_layer_sizes)
        if isinstance(PreTrained_net, nn.DataParallel):
            PreTrained_net = PreTrained_net.module

        self.MLPs = nn.ModuleList([MLP(layer_sizes) for _ in range(num_fields)]) 
        self.MLP_Final = MLP(Final_layer_sizes)

        self.PreTrained_net = PreTrained_net
        self.MapperNet = PreTrained_net.MapperNet
        self.Field_AttentionLayers = PreTrained_net.Field_AttentionLayers
        self.FinalMerge = PreTrained_net.FinalMerge
        self.PosNet = PreTrained_net.PosNet

    def standardize_features(self, latent_vectors, mean, std):
        epsilon = 1e-8
        std = torch.clamp(std, min=epsilon)
        
        standardized_vectors = (latent_vectors - mean) / std
        return standardized_vectors

    def unstandardize_features(self, standardized_vectors, mean, std):
        return standardized_vectors * std + mean

    def forward(self, Yin, Gin, num_heads, field_idx, mean_tensors, std_tensors):   # Yin and Gin are the sparse measurements of one field

        Base_Y = self.PosNet(Yin)   #   [n_batch, np_selected, n_dim -> n_base]
        Gin_Y =  torch.cat((Base_Y, Gin), dim=2)    #   [n_batch, np_selected, n_base + 1]

        #____________________________(1) PRE-TRAINED ENCODERs_____________________________________

        field_info = self.MapperNet[field_idx](Gin_Y)  #   [n_batch, np_selected, n_field_info], the UP-LIFTING net in each encoder will "take a look"
        for layer in self.Field_AttentionLayers:    #   And then, the Transformer in each encoder will re-organize the information
            field_info = layer(field_info, field_idx, num_heads)
        
        compressed_info = field_info.mean(dim=1)    #   [n_batch, n_field_info]
        STD_compressed_info = self.standardize_features(compressed_info, mean_tensors[f'field_info_{self.field_names[field_idx]}'], std_tensors[f'field_info_{self.field_names[field_idx]}'])

        #____________________________(2) Standardization and Correction______________________________
        
        Predicted_Features_list = []
        for id in range(self.n_fields):
            Feature_id = self.MLPs[id](STD_compressed_info) 
            Predicted_Features_list.append(Feature_id)
        Predicted_Features = torch.stack(Predicted_Features_list, dim=-1)   # This is supposed to be the standardized features

        UnSTD_compressed_info_list = []
        for id in range(self.n_fields):  
            UnSTD_compressed_info = self.unstandardize_features(Predicted_Features[:, :, id], mean_tensors[f'field_info_{self.field_names[id]}'], std_tensors[f'field_info_{self.field_names[id]}'])
            UnSTD_compressed_info_list.append(UnSTD_compressed_info)
        UnSTD_compressed_info_ALL = torch.stack(UnSTD_compressed_info_list, dim=-1)

        #____________________________(3) Final merge and Correction________________________________
        Global_Unified_U = self.FinalMerge(UnSTD_compressed_info_ALL, -1)
        STD_Global_Unified_U = self.standardize_features(Global_Unified_U, mean_tensors[f'Unified'], std_tensors[f'Unified'])
        U_Unified = self.MLP_Final(STD_Global_Unified_U) + STD_Global_Unified_U  # Add a ResNet. The field_idx(th) mlp is used to map from the merged_unify to final output 

        return Predicted_Features, U_Unified

# To recover the unified latent feature from single-field sparse measurements
class Direct_SensorToFeature(nn.Module):
    def __init__(self, layer_sizes):
        super(Direct_SensorToFeature, self).__init__()
        self.MLP = MLP(layer_sizes)

    def forward(self, Yin, Gin):
        Gin_flatten = Gin.squeeze(-1)

        Predicted_Feature = self.MLP(Gin_flatten)
    
        return Predicted_Feature

class Direct_SensorToField(nn.Module):
    def __init__(self, n_cond, n_sensors, n_base):
        print("n_sensors is ", n_sensors)
        print("n_base is ", n_base)

        super().__init__()

        # Channels to process the field_info for different fields, including temperature field
        self.PosNet = TrunkNet(layer_size=[2, 50, 50, 50, n_base])

        self.field_nets = nn.ModuleDict({
            'T':   BranchNet([n_sensors, 50, 50, n_base]),
            'P':   BranchNet([n_sensors, 50, 50, n_base]),
            'Vx':  BranchNet([n_sensors, 50, 50, n_base]),
            'Vy':  BranchNet([n_sensors, 50, 50, n_base]),
            'O2':  BranchNet([n_sensors, 50, 50, n_base]),
            'CO2': BranchNet([n_sensors, 50, 50, n_base]),
            'H2O': BranchNet([n_sensors, 50, 50, n_base]),
            'CO':  BranchNet([n_sensors, 50, 50, n_base]),
            'H2':  BranchNet([n_sensors, 50, 50, n_base])
        })

    def forward(self, U, Y, Gin):

        Gin_reshape = Gin.squeeze(2)
        baseF = self.PosNet(Y)
        field_outputs = {}
        for field_name, field_net in self.field_nets.items():
            coef = field_net(Gin_reshape)

            combine = coef * baseF
            field_output = torch.sum(combine, dim=2, keepdim=True)
            field_outputs[field_name] = field_output
                 
        # Stack all field outputs
        Gout = torch.cat(list(field_outputs.values()), dim=-1)
        # Reshape the output if necessary to match the desired shape
        Gout = Gout.view(U.size(0), -1, len(self.field_nets)) 

        return Gout


