
import torch
from dataclasses import dataclass

@dataclass
class DataSplit:
	U_train  : torch.Tensor
	U_test   : torch.Tensor
	Y_train  : torch.Tensor
	Y_test   : torch.Tensor
	G_train  : torch.Tensor
	G_test   : torch.Tensor

@dataclass
class DataSplit_F:
	U_train  		 : torch.Tensor
	U_test   		 : torch.Tensor
	Y_train  		 : torch.Tensor
	Gin_train  		 : torch.Tensor
	Yin_train   	 : torch.Tensor
	LR_T_train  	 : torch.Tensor
	LR_P_train  	 : torch.Tensor
	LR_Vx_train  	 : torch.Tensor
	LR_Vy_train  	 : torch.Tensor
	LR_O2_train  	 : torch.Tensor
	LR_CO2_train  	 : torch.Tensor
	LR_H2O_train  	 : torch.Tensor
	LR_CO_train		 : torch.Tensor
	LR_H2_train		 : torch.Tensor
	LR_Unified_train : torch.Tensor

	Y_test   		 : torch.Tensor
	G_train  		 : torch.Tensor
	G_test   		 : torch.Tensor
	Gin_test  		 : torch.Tensor
	Yin_test   	 	 : torch.Tensor
	LR_T_test  	 	 : torch.Tensor
	LR_P_test  	 	 : torch.Tensor
	LR_Vx_test  	 : torch.Tensor
	LR_Vy_test  	 : torch.Tensor
	LR_O2_test  	 : torch.Tensor
	LR_CO2_test  	 : torch.Tensor
	LR_H2O_test  	 : torch.Tensor
	LR_CO_test		 : torch.Tensor
	LR_H2_test		 : torch.Tensor
	LR_Unified_test  : torch.Tensor