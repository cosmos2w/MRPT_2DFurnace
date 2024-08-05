
import pickle
import torch 
import numpy as np 

from torch import nn 
from constant import DataSplit 
from torch.nn import functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU 可用")
else:
    device = torch.device("cpu")
    print("GPU 不可用，将使用 CPU")

# torch.cuda.set_device(2)
device_ids = [0, 1, 2]
device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

class ConditionNet(nn.Module):
    def __init__(self, layer_size, activation=True):
        super(ConditionNet, self).__init__()
        self.activation = activation
        self.layers = nn.ModuleList() # 用于存储所有的层
        for i in range(len(layer_size) - 1):
            # 添加一个线性层
            self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1], bias=True))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = torch.tanh(x)
                    # x = torch.relu(x)

        if self.activation:
            x = torch.sigmoid(x)
        x = x.unsqueeze(1)
        return x

class PositionNet(nn.Module):
    def __init__(self, layer_size, activation=True):
        super(PositionNet, self).__init__()
        self.activation = activation
        self.layers = nn.ModuleList() # 用于存储所有的层
        for i in range(len(layer_size) - 1):
            # 添加一个线性层
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
                    # x = torch.relu(x)

        if self.activation:
            x = torch.sigmoid(x)
        x = x.reshape(n_batch, n_point, -1)
        return x

class SubBranchNet(nn.Module):
    def __init__(self, layer_size, activation=True):
        super(SubBranchNet, self).__init__()
        self.activation = activation
        self.layers = nn.ModuleList() # 用于存储所有的层
        for i in range(len(layer_size) - 1):
            # 添加一个线性层
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

class GlobalAttentionPooling(nn.Module):
    def __init__(self, seq_depth):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.randn(seq_depth, 1))

    def forward(self, x):
        # x shape: [n_batch, seq_length, seq_depth]
        scores = torch.matmul(x, self.attention_weights)  # [n_batch, seq_length, 1]
        attention_weights = F.softmax(scores, dim=1)
        weighted_average = torch.sum(x * attention_weights, dim=1)  # [n_batch, seq_depth]
        return weighted_average

# Define the TransformerLayer as a separate class
class TransformerLayer(nn.Module):
    def __init__(self, n_field_info, num_heads,dropout_rate = 0.10):
        super().__init__()
        self.Qnet_MH = nn.Linear(n_field_info, n_field_info * num_heads)
        self.Knet_MH = nn.Linear(n_field_info, n_field_info * num_heads)
        self.Vnet_MH = nn.Linear(n_field_info, n_field_info * num_heads)
        self.output_linear = nn.Linear(n_field_info * num_heads, n_field_info)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(n_field_info)

    def forward(self, field_info, num_heads):
        # print("field_info.shape is", field_info.shape)
        # Compute Q, K, V matrices from the field_info using the respective linear layers
        Q = self.Qnet_MH(field_info)
        K = self.Knet_MH(field_info)
        V = self.Vnet_MH(field_info)

        # Split the matrices for multi-heads and reshape for batched matrix multiplication
        batch_size, seq_length, depth = Q.size()
        Q = Q.view(batch_size, seq_length, num_heads, depth // num_heads).transpose(1, 2)
        K = K.view(batch_size, seq_length, num_heads, depth // num_heads).transpose(1, 2)
        V = V.view(batch_size, seq_length, num_heads, depth // num_heads).transpose(1, 2)

        # Scaled dot-product attention for each head
        QK = torch.matmul(Q, K.transpose(-2, -1))
        attention_scores = F.softmax(QK / ((depth // num_heads) ** 0.5), dim=-1)
        weighted_values = torch.matmul(attention_scores, V)

        # Concatenate heads and reshape
        weighted_values = weighted_values.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        weighted_values = self.dropout(weighted_values) # Apply dropout to the attention output
        # Mix the information from all heads and reduce dimensionality
        output = self.output_linear(weighted_values)
        
        residual = field_info + output # Add the residual connection (input added to the output of the attention mechanism)
        output = self.norm(residual) # Apply layer normalization
        
        return output

class MultiField_TransformerLayer(nn.Module):
    def __init__(self, n_field_info, num_heads, num_fields, dropout_rate = 0.10):
        super().__init__()
        self.num_fields = num_fields
        self.Qnet_MH_list = nn.ModuleList([nn.Linear(n_field_info, n_field_info * num_heads) for _ in range(num_fields)])
        self.Knet_MH_list = nn.ModuleList([nn.Linear(n_field_info, n_field_info * num_heads) for _ in range(num_fields)])
        self.Vnet_MH_list = nn.ModuleList([nn.Linear(n_field_info, n_field_info * num_heads) for _ in range(num_fields)])
        self.output_linear_list = nn.ModuleList([nn.Linear(n_field_info * num_heads, n_field_info) for _ in range(num_fields)])
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(n_field_info)
        self.norms = nn.ModuleList([nn.LayerNorm(n_field_info) for _ in range(num_fields)])
        # self.norm2 = nn.LayerNorm(n_field_info)
        self.norm2s = nn.ModuleList([nn.LayerNorm(n_field_info) for _ in range(num_fields)])
        # self.mlp = MLP(layer_size=[n_field_info, 2 * n_field_info, n_field_info])
        self.mlps = nn.ModuleList([MLP(layer_size=[n_field_info, 2 * n_field_info, n_field_info]) for _ in range(num_fields)])

    def forward(self, field_info, field_idx, num_heads):
        # print("field_info.shape is", field_info.shape)
        batch_size, seq_length, _ = field_info.size()
        # outputs = []

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
        
        residual = field_info + output  # Add the residual connection (input added to the output of the attention mechanism)
        output = self.norms[field_idx](residual)
        
        # output = self.norm(residual)  # Apply layer normalization
        
        # norm = self.norms[field_idx](residual)  # Apply layer normalization
        # mlp_output = self.mlps[field_idx](norm)
        # output = self.norm2s[field_idx](mlp_output + norm)

        return output

class SelfAttention(nn.Module): 
    def __init__(self, n_field_info, n_fields, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
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
        # [batch_size, n_field_info, n_field_info] x [batch_size, n_field_info, n_field] -> [batch_size, n_field_info, n_field]
        unified_feature_vector = torch.bmm(attention_weights, V)
        output = unified_feature_vector.mean(dim=2)

        # Residual connection
        residual = U.mean(dim=2)  # Reducing along the n_field dimension to match output shape

        if field_idx > 0:
            Self_feature = U[:, :, field_idx] # extract the latent feature to reconstruct the field_idx(th) field extracted from itself
            output += Self_feature
        else:
            output += residual
        # output = self.norm(output + residual)
        
        # norm = self.norm(output + residual)
        # mlp_output = self.mlp(norm)
        # output = self.norm2(mlp_output + norm)

        return output

class SelfAttention_Ex0(nn.Module): 
    def __init__(self, n_field_info, n_fields, dropout_rate=0.1):
        super(SelfAttention_Ex0, self).__init__()
        self.n_field_info = n_field_info
        self.n_fields = n_fields

        self.query = nn.Linear(n_fields-3, n_field_info)
        self.key =   nn.Linear(n_fields-3, n_field_info)
        self.value = nn.Linear(n_fields-3, n_field_info)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.mlp = MLP(layer_size=[n_field_info, 2 * n_field_info, n_field_info])
        self.norm = nn.LayerNorm(n_field_info)
        self.norm2 = nn.LayerNorm(n_field_info)

    def forward(self, U, fields_to_exclude):

        fields_to_include = [i for i in range(self.n_fields) if i not in fields_to_exclude]
        # print(f'fields_to_include is ', fields_to_include)
        
        U_selected = torch.index_select(U, dim=-1, index=torch.tensor(fields_to_include, device=U.device))

        Q = self.query(U_selected)
        K = self.key(U_selected)
        V = self.value(U_selected)

        # Calculate attention scores
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.n_fields ** 0.5)
        attention_weights = self.softmax(attention_scores)
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights
        # [batch_size, n_field_info, n_field_info] x [batch_size, n_field_info, n_field] -> [batch_size, n_field_info, n_field]
        unified_feature_vector = torch.bmm(attention_weights, V)
        output = unified_feature_vector.mean(dim=2)

        # Residual connection
        residual = U_selected.mean(dim=2)  # Reducing along the n_field dimension to match output shape
        output += residual

        return output

class SelfAttention_EX(nn.Module):  #   Will exclude certain fields upon unification
    def __init__(self, n_field_info, n_fields, dropout_rate=0.1):
        super(SelfAttention_EX, self).__init__()
        self.n_field_info = n_field_info
        self.n_fields = n_fields

        # self.query = nn.Linear(n_fields, n_fields)
        # self.key = nn.Linear(n_fields, n_fields)
        # self.value = nn.Linear(n_fields, n_fields)

        self.query = nn.Linear(n_fields-3, n_field_info)    #   New roll out
        self.key =   nn.Linear(n_fields-3, n_field_info)
        self.value = nn.Linear(n_fields-3, n_field_info)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, U, field_idx, fields_to_exclude):
        # Determine the fields to include based on exclusions
        fields_to_include = [i for i in range(self.n_fields) if i not in fields_to_exclude and i != field_idx]
        # fields_to_include = [i for i in range(self.n_fields) if i not in fields_to_exclude]
        # if field_idx == 0:  print(f'fields_to_include for {field_idx} is ', fields_to_include)
        
        Self_feature = U[:, :, field_idx] # extract the latent feature to reconstruct the field_idx(th) field extracted from itself
        # if field_idx == 0:  print(f'Self_feature for {field_idx} is ', Self_feature)
        
        # Select the fields to include
        U_selected = torch.index_select(U, dim=-1, index=torch.tensor(fields_to_include, device=U.device))
        # if field_idx is 0:  print(f'U_selected.shape for {field_idx} is ', U_selected.shape)

        # Apply the attention mechanism to the selected fields
        # Q = self.query(U)
        # K = self.key(U)
        # V = self.value(U)

        Q = self.query(U_selected)  #   New roll out
        K = self.key(U_selected)
        V = self.value(U_selected)

        # Calculate attention scores
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (len(fields_to_include) ** 0.5)
        # if field_idx is 0: print(f'Before roll-out, att_score for {field_idx} is ', attention_scores)

        # Zero out attention scores for excluded fields
        # attention_mask = torch.ones_like(attention_scores)
        # attention_mask[:, :, fields_to_exclude] = 0  # Zero out scores for excluded fields in all batches and queries
        # attention_scores = attention_scores * attention_mask
        # if field_idx is 0: print(f'After roll-out, att_score for {field_idx} is ', attention_scores)

        attention_weights = self.softmax(attention_scores)
        # if field_idx is 0: print(f'After softmax, attention_weights for {field_idx} is ', attention_weights)
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights
        unified_feature_vector = torch.bmm(attention_weights, V)
        output = unified_feature_vector.mean(dim=2)

        output += Self_feature  # Add self feature back to the output
        # if field_idx == 0:  print(f'Unified_feature for {field_idx} is ', output)

        return output

class MLP(nn.Module):
    # def __init__(self, input_dim, hidden_dim, output_dim):
    #     super(MLP, self).__init__()
    #     self.fc1 = nn.Linear(input_dim, hidden_dim)
    #     self.fc2 = nn.Linear(hidden_dim, output_dim)
    #     self.dropout = nn.Dropout(0.1)
    #     self.activation = nn.ReLU()

    # def forward(self, x):
    #     x = self.fc1(x)
    #     x = self.activation(x)
    #     # x = self.dropout(x)
    #     x = self.fc2(x)
    #     return x
    # def __init__(self, layer_size, activation=True):
    #     super(MLP, self).__init__()
    #     self.activation = activation
    #     self.layers = nn.ModuleList() # 用于存储所有的层
    #     for i in range(len(layer_size) - 1):
    #         # 添加一个线性层
    #         self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1], bias=True))

    # def forward(self, x):
    #     for i, layer in enumerate(self.layers):
    #         x = layer(x)
    #         if i < len(self.layers) - 1:
    #             if self.activation:
    #                 # x = torch.tanh(x)
    #                 x = torch.relu(x)

    #     # if self.activation:
    #     #     x = torch.sigmoid(x)
    #     return x
    def __init__(self, layer_size):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList() # 用于存储所有的层
        for i in range(len(layer_size) - 1):
            # 添加一个线性层
            self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1], bias=True))
        
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()
        # self.activation = nn.Tanh()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
                x = self.dropout(x)
        return x

class MLP_SIG(nn.Module):
    def __init__(self, layer_size):
        super(MLP_SIG, self).__init__()
        self.layers = nn.ModuleList() # 用于存储所有的层
        for i in range(len(layer_size) - 1):
            # 添加一个线性层
            self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1], bias=True))
        
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
                x = self.dropout(x)

        x = torch.sigmoid(x) #  Apply Sigmoid in the last layer
        # x = torch.tanh(x) #  Apply tanh in the last layer
        return x

class DeepONet(nn.Module): 
    def __init__(self, condition_structure, position_structure, activation=True):
        super(DeepONet, self).__init__()
        self.condition_net = ConditionNet(condition_structure, activation)
        self.position_net = PositionNet(position_structure, activation)

    def forward(self, U, Y):
        coef = self.condition_net(U)
        baseF = self.position_net(Y)
        combine = coef * baseF
        out = torch.sum(combine, dim=2, keepdim=True)
        return out

class FieldToField_TransformerNet(nn.Module): 
    def __init__(self, n_field_info, n_base, num_heads, num_layers):
        print("num_heads", num_heads)
        super().__init__()
        
        # Up-lifting
        self.net_Y_Gin = PositionNet(layer_size=[ n_base + 1 , 60, 60, n_field_info])

        # Create multiple layers of the transformer
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(n_field_info, num_heads) for _ in range(num_layers)
        ])
        # Add a final LayerNorm before taking the mean
        self.final_norm = nn.LayerNorm(n_field_info)

        # Channels to process the field_info for different fields, including temperature field
        self.PosNet = PositionNet(layer_size=[2, 50, 50, 50, n_base])
        self.field_nets = nn.ModuleDict({
            'T':   ConditionNet([n_field_info, 50, 50, n_base]),
            'P':   ConditionNet([n_field_info, 50, 50, n_base]),
            'Vx':  ConditionNet([n_field_info, 50, 50, n_base]),
            'Vy':  ConditionNet([n_field_info, 50, 50, n_base]),
            'O2':  ConditionNet([n_field_info, 50, 50, n_base]),
            'CO2': ConditionNet([n_field_info, 50, 50, n_base]),
            'H2O': ConditionNet([n_field_info, 50, 50, n_base]),
            'CO':  ConditionNet([n_field_info, 50, 50, n_base]),
            'H2':  ConditionNet([n_field_info, 50, 50, n_base])
        })

    def _compress_data(self, Y, Gin, num_heads):
        # print(Y.shape)
        # print(Gin.shape)
        Gin_Y =  torch.cat((Y, Gin), dim=2) # [n_batch, n_points, n_encoding + 1]
        base_for_Gin_Y = self.net_Y_Gin(Gin_Y)

        field_info = base_for_Gin_Y
        for layer in self.transformer_layers:
            field_info = layer(field_info, num_heads)

        # After all transformer layers, perform LayerNorm (if desired)
        field_info = self.final_norm(field_info)

        # Calculate the mean across the sequence length dimension (dim=1)
        compressed_info = field_info.mean(dim=1)

        return compressed_info

    def forward(self, Y, Gin, num_heads):

        baseF = self.PosNet(Y)
        field_info = self._compress_data(baseF, Gin, num_heads)
        # print(baseF.shape)

        # Compute outputs for each field using the corresponding network
        field_outputs = {}
        for field_name, field_net in self.field_nets.items():
            coef = field_net(field_info)
            combine = coef * baseF
            field_output = torch.sum(combine, dim=2, keepdim=True)
            field_outputs[field_name] = field_output

        # Stack all field outputs
        Gout = torch.cat(list(field_outputs.values()), dim=-1)
        # Reshape the output if necessary to match the desired shape
        Gout = Gout.view(field_info.size(0), -1, len(self.field_nets))  

        return Gout

class Self_Representation_PreTrain_Net(nn.Module): 
    def __init__(self, n_field_info, n_base, num_heads, num_layers, num_fields):
        print("num_heads", num_heads)
        print("num_fields", num_fields)
        print("n_base", n_base)
        super().__init__()

        # Up-lifting
        self.net_Y_Gins = nn.ModuleList([PositionNet(layer_size=[ n_base + 1 , 60, 60, n_field_info]) for _ in range(num_fields)])

        # Create multiple layers of the transformer
        self.transformer_layers = nn.ModuleList([
            MultiField_TransformerLayer(n_field_info, num_heads, num_fields, dropout_rate = 0.30) for _ in range(num_layers)
        ])
        # Add a final LayerNorm before taking the mean
        self.final_norm = nn.LayerNorm(n_field_info)
        
        #To combine different feature vectors
        self.UnifyAttention_layers = nn.ModuleList([
            SelfAttention(n_field_info*2, num_fields) for _ in range(num_fields)
        ])
        self.FinalMerge = SelfAttention(n_field_info, num_fields, dropout_rate = 0.05) 

        self.PosNet = PositionNet(layer_size=[2, 50, 50, 50, n_base])
        # self.Unified_field_net = ConditionNet(layer_size=[n_field_info, 50, 50, n_base])
        self.field_nets = nn.ModuleDict({
            'T':   ConditionNet([n_field_info, 50, 50, n_base]),
            'P':   ConditionNet([n_field_info, 50, 50, n_base]),
            'Vx':  ConditionNet([n_field_info, 50, 50, n_base]),
            'Vy':  ConditionNet([n_field_info, 50, 50, n_base]),
            'O2':  ConditionNet([n_field_info, 50, 50, n_base]),
            'CO2': ConditionNet([n_field_info, 50, 50, n_base]),
            'H2O': ConditionNet([n_field_info, 50, 50, n_base]),
            'CO':  ConditionNet([n_field_info, 50, 50, n_base]),
            'H2':  ConditionNet([n_field_info, 50, 50, n_base])
        })

    def _compress_data(self, Y, Gin, num_heads):
        n_fields = Gin.shape[-1]
        # print('n_fields is ', n_fields)

        compressed_info_list = []
        for field_idx in range(n_fields):
            
            Gin_Y =  torch.cat((Y, Gin[:, :, field_idx:field_idx+1]), dim=2) # [n_batch, n_points, n_encoding + 1] * n_fields            
            base_for_Gin_Y = self.net_Y_Gins[field_idx](Gin_Y)
            
            field_info = base_for_Gin_Y         
            for layer in self.transformer_layers:
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

class Mutual_Representation_PreTrain_Net_SingleMerge(nn.Module): 
    def __init__(self, n_field_info, n_base, num_heads, num_layers, num_fields=9):
        print("num_heads is", num_heads)
        print("num_fields is", num_fields)
        print("n_field_info is", n_field_info)
        super().__init__()

        # Up-lifting
        self.net_Y_Gins = nn.ModuleList([PositionNet(layer_size=[ n_base + 1 , 50, 50, n_field_info]) for _ in range(num_fields)])
        # self.net_Y_Gins = nn.ModuleList([MLP(layer_size=[ n_base + 1 , 60, 60, 60, n_field_info]) for _ in range(num_fields)])
        # self.net_Y_Gins = nn.ModuleList([MLP_SIG(layer_size=[ n_base + 1 , 100, 100, 100, n_field_info]) for _ in range(num_fields)])

        # Create multiple layers of the transformer for different fields
        self.transformer_layers = nn.ModuleList([
            MultiField_TransformerLayer(n_field_info, num_heads, num_fields, dropout_rate = 0.40) for _ in range(num_layers)
        ])
        # Add a final LayerNorm before taking the mean
        self.final_norm = nn.LayerNorm(n_field_info)
        #   Use attention-based pooling technique
        # self.attention_pool = GlobalAttentionPooling(n_field_info)
        self.attention_pools = nn.ModuleList([
            GlobalAttentionPooling(n_field_info) for _ in range(num_fields)
        ])
        
        #To combine different feature vectors
        self.UnifyAttention_layers = nn.ModuleList([
            SelfAttention(n_field_info, num_fields) for _ in range(num_fields)
        ])
        
        # self.FinalMerge = SelfAttention(n_field_info, num_fields, dropout_rate = 0.10) 
        self.FinalMerge = SelfAttention_Ex0(n_field_info, num_fields, dropout_rate = 0.10)  #   This is for roll-out

        self.MLPs = nn.ModuleList([MLP(layer_size=[n_field_info, 60, n_field_info]) for _ in range(num_fields)])

        # Channels to process the field_info for different fields, including temperature field
        self.PosNet = PositionNet(layer_size=[2, 60, 60, 60, n_base])
        self.PosNets = nn.ModuleDict({
            'T':   PositionNet([2, 50, 50, 50, n_base]),
            'P':   PositionNet([2, 50, 50, 50, n_base]),
            'Vx':  PositionNet([2, 50, 50, 50, n_base]),
            'Vy':  PositionNet([2, 50, 50, 50, n_base]),
            'O2':  PositionNet([2, 50, 50, 50, n_base]),
            'CO2': PositionNet([2, 50, 50, 50, n_base]),
            'H2O': PositionNet([2, 50, 50, 50, n_base]),
            'CO':  PositionNet([2, 50, 50, 50, n_base]),
            'H2':  PositionNet([2, 50, 50, 50, n_base])
        })
        self.field_nets = nn.ModuleDict({
            'T':   ConditionNet([n_field_info, 50, 50, n_base]),
            'P':   ConditionNet([n_field_info, 50, 50, n_base]),
            'Vx':  ConditionNet([n_field_info, 50, 50, n_base]),
            'Vy':  ConditionNet([n_field_info, 50, 50, n_base]),
            'O2':  ConditionNet([n_field_info, 50, 50, n_base]),
            'CO2': ConditionNet([n_field_info, 50, 50, n_base]),
            'H2O': ConditionNet([n_field_info, 50, 50, n_base]),
            'CO':  ConditionNet([n_field_info, 50, 50, n_base]),
            'H2':  ConditionNet([n_field_info, 50, 50, n_base])
        })

    def _compress_data(self, Y, Gin, num_heads): # Y = baseF here
        n_fields = Gin.shape[-1]

        # Concatenate Y with all fields in Gin
        # Gin_Y = torch.cat((Y, Gin), dim=2)  # [n_batch, n_points, n_encoding + n_fields]

        compressed_info_list = []
        for field_idx in range(n_fields):
            
            Gin_Y =  torch.cat((Y, Gin[:, :, field_idx:field_idx+1]), dim=2) # [n_batch, n_points, n_encoding + 1] * n_fields
            
            base_for_Gin_Y = self.net_Y_Gins[field_idx](Gin_Y)
            field_info = base_for_Gin_Y
            
            for layer in self.transformer_layers:
                field_info = layer(field_info, field_idx, num_heads)
            # After all transformer layers, perform LayerNorm (if desired)
            # field_info = self.final_norm(field_info)
            
            # # Calculate the mean across the sequence length dimension (dim=1)
            # compressed_mean = field_info.mean(dim=1)
            # # Apply max pooling across the dimension of selected points
            # compressed_max, _ = field_info.max(dim=1)
            # # compressed_info = torch.cat((compressed_mean, compressed_max), dim=1)
            # compressed_info = compressed_mean + compressed_max

            #   Use attention-based pooling technique
            # compressed_info = self.attention_pools[field_idx](field_info)

            compressed_info = field_info.mean(dim=1)
            compressed_info_list.append(compressed_info)
                   
        compressed_info_ALL = torch.stack(compressed_info_list, dim=-1)
        return compressed_info_ALL

    def UnifyAttention(self, field_info):
        U_sliced_transformed = []
        
        for field_idx in range(field_info.shape[-1]):
            U_Unified = self.UnifyAttention_layers[field_idx](field_info)
            # U_Unified = self.MLPs[field_idx](U_Unified)           
            U_sliced_transformed.append(U_Unified)
        
        # Stack the transformed slices to form a new tensor
        U_transformed = torch.stack(U_sliced_transformed, dim=-1)
        Output = field_info + U_transformed  # Add the residual connection
        
        # return U_transformed
        return Output

    def mlp_transform(self, field_info):
        # Apply MLP to each slice and store the results in a list
        U_sliced_transformed = []
        for field_idx in range(field_info.shape[-1]):
            U_sliced = field_info[:, :, field_idx]
            U_sliced_transformed.append( self.MLPs[field_idx](U_sliced) )
        # Stack the transformed slices to form a new tensor
        U_transformed = torch.stack(U_sliced_transformed, dim=-1)
        return U_transformed

    def forward(self, cond, Y, Gin, att_index, num_heads):
        n_batch = cond.shape[0]
        n_fields = Gin.shape[-1]
        baseF = self.PosNet(Y)

        att_index = att_index[0]

        # field_info = self._compress_data(Y, Gin, num_heads)
        field_info = self._compress_data(baseF, Gin, num_heads)

        fields_to_exclude = att_index[-3:].tolist()  # Determine fields to exclude based on performance;
        # print(f'fields_to_exclude of is', fields_to_exclude)

        U = field_info
        U_Unified = self.FinalMerge(U, fields_to_exclude)

        Gout_list = []
        for field_idx in range(n_fields + 1): # Includes an extra iteration for the unified feature
            # Determine the input tensor: U_sliced for fields or U_Unified for unified feature
            U_input = U[:, :, field_idx] if field_idx < n_fields else U_Unified
            # U_input = U_transformed[:, :, field_idx] if field_idx < n_fields else U_Unified
            
            # Compute outputs for each field using the corresponding network
            field_outputs = []
            for field_name, field_net in self.field_nets.items(): # Generate all the fields
                coef = field_net(U_input)
                # baseF = self.PosNets[field_name](Y)  # This computes the base functions for the field
                combine = coef * baseF
                field_output = torch.sum(combine, dim=2, keepdim=True)
                field_outputs.append(field_output)
            
            # Stack all field outputs and reshape
            Gout = torch.cat(field_outputs, dim=-1)
            Gout = Gout.view(U.size(0), -1, len(self.field_nets))  # Assuming the second dimension is correctly sized
            Gout_list.append(Gout)

        return Gout_list

class Mutual_Representation_PreTrain_Net(nn.Module): # Pre-training F2F task via mutual representation
    def __init__(self, n_field_info, n_base, num_heads, num_layers, num_fields):
        print("num_heads is", num_heads)
        print("num_fields is", num_fields)
        print("n_field_info is", n_field_info)
        super().__init__()

        # Up-lifting
        self.net_Y_Gins = nn.ModuleList([PositionNet(layer_size=[ n_base + 1 , 50, 50, n_field_info ]) for _ in range(num_fields)])

        # Create multiple layers of MH transformers for encoders towards different fields
        self.transformer_layers = nn.ModuleList([
            MultiField_TransformerLayer(n_field_info, num_heads, num_fields, dropout_rate = 0.40) for _ in range(num_layers)
        ])
        
        self.FieldMerges = nn.ModuleList([
            SelfAttention(n_field_info, num_fields, dropout_rate = 0.10) for _ in range(num_fields)
        ])
        self.FinalMerge = SelfAttention(n_field_info, num_fields, dropout_rate = 0.05) 

        # Channels to process the field_info for different fields, including temperature field
        self.PosNet = PositionNet(layer_size=[2, 50, 50, 50, n_base])
        self.field_names = ['T', 'P', 'Vx', 'Vy', 'O2', 'CO2', 'H2O', 'CO', 'H2']
        self.field_nets = nn.ModuleDict({
            'T':   ConditionNet([n_field_info, 50, 50, n_base]),
            'P':   ConditionNet([n_field_info, 50, 50, n_base]),
            'Vx':  ConditionNet([n_field_info, 50, 50, n_base]),
            'Vy':  ConditionNet([n_field_info, 50, 50, n_base]),
            'O2':  ConditionNet([n_field_info, 50, 50, n_base]),
            'CO2': ConditionNet([n_field_info, 50, 50, n_base]),
            'H2O': ConditionNet([n_field_info, 50, 50, n_base]),
            'CO':  ConditionNet([n_field_info, 50, 50, n_base]),
            'H2':  ConditionNet([n_field_info, 50, 50, n_base])
        })

    def _compress_data(self, Y, Gin, field_idx, num_heads): # Y = baseF here
        n_fields = Gin.shape[-1]
        compressed_info_list = []
        for id in range(n_fields):  #   For all fields in Gin, pass them through the field_idx(th) Encoder, and obtain LRs towards the field_idx(th) field
            
            Gin_Y =  torch.cat((Y, Gin[:, :, id:id+1]), dim=2) # [n_batch, n_points, n_encoding + 1] * n_fields

            # The field_idx(th) Encoder:             
            field_info = self.net_Y_Gins[field_idx](Gin_Y)            
            for layer in self.transformer_layers:
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
            # print('field_name is', field_name)
            field_info = self._compress_data(baseF, Gin, field_idx, num_heads) # [n_batch, n_field_info, n_fields]: The latent representations from one Encoder towards the field_idx(th) field
            # print('field_info.shape is', field_info.shape)
            U_Unified = self.FieldMerges[field_idx](field_info, field_idx) #   [n_batch, n_field_info]: Unified latent representations from one Encoder towards the field_idx(th) field
            # print('U_Unified.shape is', U_Unified.shape)

            field_outputs = []
            for id in range(n_fields +1):
                U_input = field_info[:, :, id] if id < n_fields else U_Unified
                
                coef = self.field_nets[field_name](U_input)  #   Branch net for the field_idx(th) field
                combine = coef * baseF
                field_output = torch.sum(combine, dim=2, keepdim=True)  #   Reconstructed field_idx(th) field from the latent representation of the id(th) field or Unified
                field_outputs.append(field_output)
            
            # Stack all field outputs and reshape 
            Gout = torch.cat(field_outputs, dim=-1) #   All the field_idx(th) field results from different input fields, [n_batch, n_points, n_fields + 1]
            # print('Gout.shape is', Gout.shape)
            
            U_Unified_list.append(U_Unified)
            Gout_list.append(Gout)
        
        Global_Unified_U = torch.stack(U_Unified_list, dim=2) #   [n_batch, n_field_info, n_fields]
        # print('Global_Unified_U.shape is', Global_Unified_U.shape)
        Global_Unified_U = self.FinalMerge(Global_Unified_U, field_idx = -1)   #   [n_batch, n_field_info]

        Global_Unified_field_outputs = []
        for field_name, field_net in self.field_nets.items(): # Generate all the fields
            coef = field_net(Global_Unified_U)
            # baseF = self.PosNets[field_name](Y)  # This computes the base functions for the field
            combine = coef * baseF
            Global_Unified_field_output = torch.sum(combine, dim=2, keepdim=True)
            Global_Unified_field_outputs.append(Global_Unified_field_output)
        Global_Unified_Gout = torch.cat(Global_Unified_field_outputs, dim=-1) #   All the field_idx(th) field results from Global_Unified_U

        return Gout_list, Global_Unified_Gout

# To predict unseen variables based on pre-trained net
class Mutual_MultiEn_MultiDe_FineTune_Net(nn.Module):
    def __init__(self, n_inputF, n_field_info, n_base, PreTrained_net):
        super(Mutual_MultiEn_MultiDe_FineTune_Net, self).__init__()
        
        if isinstance(PreTrained_net, nn.DataParallel):
            PreTrained_net = PreTrained_net.module
        
        # The ConditionNet in decoder for downstream unseen field to be trained
        self.field_net = ConditionNet([n_field_info, 50, 50, n_base]) 

        # The PositionNet will be used and trained if mode == False
        self.CondNet_Backup = ConditionNet([n_inputF, 50, 50, n_base])
        self.PosNet_Backup = PositionNet([2, 50, 50, 50, n_base])
        
        self.PreTrained_net = PreTrained_net
        self._compress_data = PreTrained_net._compress_data
        self.PosNet = PreTrained_net.PosNet
        self.FieldMerges = PreTrained_net.FieldMerges
        self.FinalMerge = PreTrained_net.FinalMerge

    def forward(self, U, Y, G_PreTrain, num_heads, att_index, mode):
        if mode is True: # True means the reconstruction will proceed based on the pre-trained net 

            with torch.no_grad(): 
                baseF = self.PosNet(Y)
                att_index = att_index[0]

                n_fields = G_PreTrain.shape[-1]
                # print('G_PreTrain.shape is ', G_PreTrain.shape)
                # print('att_index is', att_index)
                U_Unified_list = []
                for field_idx in range(n_fields):
                    field_info = self._compress_data(baseF, G_PreTrain, field_idx, num_heads) 
                    
                    fields_to_exclude = att_index[field_idx, -2:].tolist()  
                    if field_idx in fields_to_exclude:
                        # If yes, extend the exclusion to include one more index
                        fields_to_exclude = att_index[field_idx, -3:].tolist()  # Now taking the last four indices
                    # print('fields_to_exclude is ',fields_to_exclude)

                    U_Unified = self.FieldMerges[field_idx](field_info, field_idx, fields_to_exclude)
                    U_Unified_list.append(U_Unified)

                Global_Unified_U = torch.stack(U_Unified_list, dim=2) 
                Global_Unified_U = self.FinalMerge(Global_Unified_U)   
            
            coef = self.field_net(Global_Unified_U)
        else:
            baseF = self.PosNet_Backup(Y)
            coef = self.CondNet_Backup(U)

        combine = coef * baseF
        field_output = torch.sum(combine, dim=2, keepdim=True)
        # print('field_output.shape is ', field_output.shape)

        return field_output

# To recover the unified latent feature from single-field sparse measurements
class Direct_SensorToFeature(nn.Module):
    def __init__(self, layer_sizes):
        super(Direct_SensorToFeature, self).__init__()

        self.MLP = MLP(layer_sizes)

    def forward(self, Yin, Gin):
        # print("Gin.shape is ", Gin.shape)
        Gin_flatten = Gin.squeeze(-1)

        Predicted_Feature = self.MLP(Gin_flatten)
    
        return Predicted_Feature


# To recover the unified latent feature from single-field sparse measurements
class Mutual_SensorToFeature(nn.Module):
    def __init__(self, layer_sizes, PreTrained_net):
        super(Mutual_SensorToFeature, self).__init__()

        # print('layer_sizes is ', layer_sizes)
        if isinstance(PreTrained_net, nn.DataParallel):
            PreTrained_net = PreTrained_net.module

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        self.PreTrained_net = PreTrained_net
        self.net_Y_Gins = PreTrained_net.net_Y_Gins
        self.transformer_layers = PreTrained_net.transformer_layers
        self.final_norm = PreTrained_net.final_norm
        self.attention = PreTrained_net.attention
        
    def normalize_data(self, data, min_val, max_val):
        # Normalize data to [0, 1]
        normalized_data = (data - min_val) / (max_val - min_val)
        
        # sqrt_normalized_data = torch.sqrt(normalized_data) # Take the square root of the normalized data
        # return sqrt_normalized_data
        return normalized_data

    def forward(self, Yin, Gin, field_idx, num_heads, min_val, max_val):

        Gin_Y =  torch.cat((Yin, Gin), dim=2)
        base_for_Gin_Y = self.net_Y_Gins[field_idx](Gin_Y)
        field_info = self.transformer_layers[field_idx](base_for_Gin_Y, field_idx, num_heads)
        field_info = self.final_norm(field_info)
        compressed_info = field_info.mean(dim=1)
        # print('compressed_info.shape is ', compressed_info.shape)

        # Normalize compressed_info
        # min_val, max_val = compressed_info.min(), compressed_info.max()
        compressed_info = self.normalize_data(compressed_info, min_val, max_val)

        Predicted_Feature = compressed_info
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:  # Last layer without activation function
                Predicted_Feature = layer(Predicted_Feature)
            else:
                Predicted_Feature = torch.relu(layer(Predicted_Feature))
    
        return Predicted_Feature

# Using F2F inter-inference to recover the unified latent feature from single-field sparse measurements
class Mutual_SensorToFeature_InterInference(nn.Module):
    def __init__(self, layer_sizes, PreTrained_net, num_fields = 9):
        super(Mutual_SensorToFeature_InterInference, self).__init__()
        self.n_fields = num_fields

        print('layer_sizes of MLP is ', layer_sizes)
        if isinstance(PreTrained_net, nn.DataParallel):
            PreTrained_net = PreTrained_net.module

        self.MLPs = nn.ModuleList([MLP(layer_sizes) for _ in range(num_fields)]) 
        self.MLP_Final = MLP([36, 300, 300, 36])

        self.PreTrained_net = PreTrained_net
        self.net_Y_Gins = PreTrained_net.net_Y_Gins
        self.transformer_layers = PreTrained_net.transformer_layers
        self.FinalMerge = PreTrained_net.FinalMerge
        self.PosNet = PreTrained_net.PosNet
        
    def normalize_data(self, data, min_val, max_val):
        # Normalize data to [0, 1]
        # print('Input data is ', data)
        normalized_data = (data - min_val) / (max_val - min_val)
        # print('normalized data is ', normalized_data)

        # Check for negative values in the normalized data
        # if torch.any(normalized_data < 0):
        #     negative_values = normalized_data[normalized_data < 0]
        #     raise ValueError(f"Negative values found in normalized data: {negative_values}")

        sqrt_normalized_data = torch.sqrt(torch.clamp(normalized_data, min=0))
        
        return sqrt_normalized_data

    def Denormalize_data(self, normalized_data, min_val, max_val):

        Retrieved_data = normalized_data ** 2
        # print('Retrieved_data is ', Retrieved_data)
        Retrieved_data = Retrieved_data * (max_val - min_val) + min_val
        return Retrieved_data

    def forward(self, Yin, Gin, num_heads, min_val, max_val):   # Yin and Gin are the sparse measurements of one field

        # print('Inside, Yin.shape is', Yin.shape)

        Base_Y = self.PosNet(Yin)   #   [n_batch, np_selected, n_dim -> n_base]
        Gin_Y =  torch.cat((Base_Y, Gin), dim=2)    #   [n_batch, np_selected, n_base + 1]

        #____________________________(1) PRE-TRAINED ENCODERs_____________________________________
        compressed_info_list = []
        for id in range(self.n_fields):  
            field_info = self.net_Y_Gins[id](Gin_Y)  #   [n_batch, np_selected, n_field_info], the UP-LIFTING net in each encoder will "take a look"
            for layer in self.transformer_layers:    #   And then, the Transformer in each encoder will re-organize the information
                field_info = layer(field_info, id, num_heads)
            compressed_info = field_info.mean(dim=1)    #   [n_batch, n_field_info]
            compressed_info_list.append(compressed_info)
        compressed_info_ALL = torch.stack(compressed_info_list, dim=-1) # [n_batch, n_field_info, n_fields]: The latent representations from one field out of all Encoders
        # print('compressed_info_ALL is ', compressed_info_ALL)

        #____________________________(2) Normalization and Correction______________________________
        Norm_compressed_info = self.normalize_data(compressed_info_ALL, min_val, max_val)

        # New tensor by concatenating the last dimension
        compressed_info_concat = compressed_info_ALL.reshape(Norm_compressed_info.shape[0], -1)  # [n_batch, n_field_info * n_fields]

        Predicted_Features_list = []
        for id in range(self.n_fields):
            Feature_id = self.MLPs[id](compressed_info_concat) + Norm_compressed_info[:, :, id]  #   MLPs will map the Concatenated APPROXIMATE latent representations to the accurate UNIFEID one from the id(th) encoder

            Predicted_Features_list.append(Feature_id)
        Predicted_Features = torch.stack(Predicted_Features_list, dim=-1) # These will be the NORMALIZED latent representations for all fields

        # Predicted_Features = Norm_compressed_info

        #____________________________(3) Final merge and Correction________________________________
        DeNorm_Predicted_Features_list = []
        for id in range(self.n_fields):
            DeNorm_Feature_id = self.Denormalize_data(Predicted_Features[:, :, id], min_val, max_val)
            DeNorm_Predicted_Features_list.append(DeNorm_Feature_id)
        DeNorm_Predicted_Features = torch.stack(DeNorm_Predicted_Features_list, dim=-1)
        # print('DeNorm_Predicted_Features is ', DeNorm_Predicted_Features)
        
        Global_Unified_U = self.FinalMerge(DeNorm_Predicted_Features, -1)
        # print('Global_Unified_U is ', Global_Unified_U)
        Norm_Global_Unified_U = self.normalize_data(Global_Unified_U, min_val, max_val)

        U_Unified = self.MLP_Final(Norm_Global_Unified_U) + Norm_Global_Unified_U  # Add a ResNet. The field_idx(th) mlp is used to map from the merged_unify to final output 
    
        return Predicted_Features, U_Unified

# Perform parameter inversion task: Using F2F inter-inference to recover the unified latent feature from single-field sparse measurements
class New_Mutual_SensorToParameter(nn.Module):
    def __init__(self, layer_sizes, PreTrained_net, num_fields = 9):
        super(New_Mutual_SensorToParameter, self).__init__()
        self.n_fields = num_fields

        print('layer_sizes of MLP is ', layer_sizes)
        if isinstance(PreTrained_net, nn.DataParallel):
            PreTrained_net = PreTrained_net.module

        self.MLPs = nn.ModuleList([MLP(layer_sizes) for _ in range(num_fields)]) 
        self.MLP_Final = MLP([36, 300, 300, 36])
        
        self.MLP_LR2Cond = MLP([36, 200, 200, 11]) # mapping the latent representation to U

        self.PreTrained_net = PreTrained_net
        self.net_Y_Gins = PreTrained_net.net_Y_Gins
        self.transformer_layers = PreTrained_net.transformer_layers
        self.FinalMerge = PreTrained_net.FinalMerge
        self.PosNet = PreTrained_net.PosNet
        
    def normalize_data(self, data, min_val, max_val):
        # Normalize data to [0, 1]
        # print('Input data is ', data)
        normalized_data = (data - min_val) / (max_val - min_val)
        # print('normalized data is ', normalized_data)

        # Check for negative values in the normalized data
        # if torch.any(normalized_data < 0):
        #     negative_values = normalized_data[normalized_data < 0]
        #     raise ValueError(f"Negative values found in normalized data: {negative_values}")

        sqrt_normalized_data = torch.sqrt(torch.clamp(normalized_data, min=0))
        
        return sqrt_normalized_data

    def Denormalize_data(self, normalized_data, min_val, max_val):

        Retrieved_data = normalized_data ** 2
        # print('Retrieved_data is ', Retrieved_data)
        Retrieved_data = Retrieved_data * (max_val - min_val) + min_val
        return Retrieved_data

    def forward(self, Yin, Gin, num_heads, min_val, max_val):   # Yin and Gin are the sparse measurements of one field

        # print('Inside, Yin.shape is', Yin.shape)

        Base_Y = self.PosNet(Yin)   #   [n_batch, np_selected, n_dim -> n_base]
        Gin_Y =  torch.cat((Base_Y, Gin), dim=2)    #   [n_batch, np_selected, n_base + 1]

        #____________________________(1) PRE-TRAINED ENCODERs_____________________________________
        compressed_info_list = []
        for id in range(self.n_fields):  
            field_info = self.net_Y_Gins[id](Gin_Y)  #   [n_batch, np_selected, n_field_info], the UP-LIFTING net in each encoder will "take a look"
            for layer in self.transformer_layers:    #   And then, the Transformer in each encoder will re-organize the information
                field_info = layer(field_info, id, num_heads)
            compressed_info = field_info.mean(dim=1)    #   [n_batch, n_field_info]
            compressed_info_list.append(compressed_info)
        compressed_info_ALL = torch.stack(compressed_info_list, dim=-1) # [n_batch, n_field_info, n_fields]: The latent representations from one field out of all Encoders
        # print('compressed_info_ALL is ', compressed_info_ALL)

        #____________________________(2) Normalization and Correction______________________________
        Norm_compressed_info = self.normalize_data(compressed_info_ALL, min_val, max_val)

        # New tensor by concatenating the last dimension
        compressed_info_concat = compressed_info_ALL.reshape(Norm_compressed_info.shape[0], -1)  # [n_batch, n_field_info * n_fields]

        Predicted_Features_list = []
        for id in range(self.n_fields):
            
            # Feature_id = self.MLPs[id](Norm_compressed_info[:, :, id])   #   MLPs will map the APPROXIMATE latent representations to the accurate UNIFEID one from the id(th) encoder

            Feature_id = self.MLPs[id](compressed_info_concat) + Norm_compressed_info[:, :, id]  #   MLPs will map the Concatenated APPROXIMATE latent representations to the accurate UNIFEID one from the id(th) encoder

            Predicted_Features_list.append(Feature_id)
        Predicted_Features = torch.stack(Predicted_Features_list, dim=-1) # These will be the NORMALIZED latent representations for all fields
        # Predicted_Features = Norm_compressed_info

        #____________________________(3) Final merge and Correction________________________________
        DeNorm_Predicted_Features_list = []
        for id in range(self.n_fields):
            DeNorm_Feature_id = self.Denormalize_data(Predicted_Features[:, :, id], min_val, max_val)
            DeNorm_Predicted_Features_list.append(DeNorm_Feature_id)
        DeNorm_Predicted_Features = torch.stack(DeNorm_Predicted_Features_list, dim=-1)
        # print('DeNorm_Predicted_Features is ', DeNorm_Predicted_Features)
        
        Global_Unified_U = self.FinalMerge(DeNorm_Predicted_Features)
        # print('Global_Unified_U is ', Global_Unified_U)
        Norm_Global_Unified_U = self.normalize_data(Global_Unified_U, min_val, max_val)

        U_Unified = self.MLP_Final(Norm_Global_Unified_U) + Norm_Global_Unified_U  # Add a ResNet. The field_idx(th) mlp is used to map from the merged_unify to final output 

        Cond = self.MLP_LR2Cond(U_Unified)
    
        return Predicted_Features, U_Unified, Cond

class Direct_SensorToField(nn.Module):
    def __init__(self, n_cond, n_sensors, n_base):
        print("n_sensors is ", n_sensors)
        print("n_base is ", n_base)

        super().__init__()

        # Channels to process the field_info for different fields, including temperature field
        self.PosNet = PositionNet(layer_size=[2, 50, 50, 50, n_base])

        self.field_nets = nn.ModuleDict({
            'T':   ConditionNet([n_sensors, 50, 50, n_base]),
            'P':   ConditionNet([n_sensors, 50, 50, n_base]),
            'Vx':  ConditionNet([n_sensors, 50, 50, n_base]),
            'Vy':  ConditionNet([n_sensors, 50, 50, n_base]),
            'O2':  ConditionNet([n_sensors, 50, 50, n_base]),
            'CO2': ConditionNet([n_sensors, 50, 50, n_base]),
            'H2O': ConditionNet([n_sensors, 50, 50, n_base]),
            'CO':  ConditionNet([n_sensors, 50, 50, n_base]),
            'H2':  ConditionNet([n_sensors, 50, 50, n_base])
        })

    def forward(self, U, Y, Gin):
        # print('Gin.shape is ', Gin.shape)
        # print('Y.shape is ', Gin.shape)
        Gin_reshape = Gin.squeeze(2)
        # print('Gin_reshape.shape is ', Gin_reshape.shape)
        baseF = self.PosNet(Y)
        field_outputs = {}
        for field_name, field_net in self.field_nets.items():
            coef = field_net(Gin_reshape)
            # coef = field_net(U)
            # baseF = self.PosNets[field_name](Y)  # This computes the base functions for the field
            combine = coef * baseF
            field_output = torch.sum(combine, dim=2, keepdim=True)
            field_outputs[field_name] = field_output
                 
        # Stack all field outputs
        Gout = torch.cat(list(field_outputs.values()), dim=-1)
        # Reshape the output if necessary to match the desired shape
        Gout = Gout.view(U.size(0), -1, len(self.field_nets)) 

        return Gout

if __name__ == '__main__':

    with open('data_split.pic', 'rb') as fp:
        data_split = pickle.load(fp)

        U = data_split.U_test
        Y = data_split.Y_test
        G = data_split.G_test

        n_inputF = U.shape[-1]
        n_pointD = Y.shape[-1]
        n_baseF = 1

    deep_onet = DeepONet(n_inputF, n_pointD, n_baseF)

    G = deep_onet(U, Y)

    print(G.shape)


