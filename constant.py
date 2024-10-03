
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

@dataclass
class DataSplit_STD:
	U_train  		 : torch.Tensor
	Y_train  		 : torch.Tensor
	G_train  		 : torch.Tensor

	Gin_train  		 : torch.Tensor
	Yin_train   	 : torch.Tensor

	LR_T_train  	 : torch.Tensor
	MEAN_T_train     : torch.Tensor
	STD_T_train  	 : torch.Tensor

	LR_P_train  	 : torch.Tensor
	MEAN_P_train     : torch.Tensor
	STD_P_train  	 : torch.Tensor

	LR_Vx_train   	 : torch.Tensor
	MEAN_Vx_train 	 : torch.Tensor
	STD_Vx_train  	 : torch.Tensor

	LR_Vy_train      : torch.Tensor
	MEAN_Vy_train    : torch.Tensor
	STD_Vy_train     : torch.Tensor
   
	LR_O2_train      : torch.Tensor
	MEAN_O2_train    : torch.Tensor
	STD_O2_train     : torch.Tensor

	LR_CO2_train     : torch.Tensor
	MEAN_CO2_train   : torch.Tensor
	STD_CO2_train    : torch.Tensor

	LR_H2O_train     : torch.Tensor
	MEAN_H2O_train   : torch.Tensor
	STD_H2O_train    : torch.Tensor

	LR_CO_train      : torch.Tensor
	MEAN_CO_train    : torch.Tensor
	STD_CO_train     : torch.Tensor

	LR_H2_train      : torch.Tensor
	MEAN_H2_train    : torch.Tensor
	STD_H2_train     : torch.Tensor

	LR_Unified_train : torch.Tensor
	MEAN_Unified_train : torch.Tensor
	STD_Unified_train  : torch.Tensor

	U_test   		 : torch.Tensor
	Y_test  		 : torch.Tensor
	G_test   		 : torch.Tensor

	Gin_test  		 : torch.Tensor
	Yin_test   	 	 : torch.Tensor

	LR_T_test  	 	 : torch.Tensor
	MEAN_T_test      : torch.Tensor
	STD_T_test  	 : torch.Tensor

	LR_P_test  	 	 : torch.Tensor
	MEAN_P_test      : torch.Tensor
	STD_P_test  	 : torch.Tensor

	LR_Vx_test   	 : torch.Tensor
	MEAN_Vx_test 	 : torch.Tensor
	STD_Vx_test  	 : torch.Tensor

	LR_Vy_test       : torch.Tensor
	MEAN_Vy_test     : torch.Tensor
	STD_Vy_test      : torch.Tensor
    
	LR_O2_test       : torch.Tensor
	MEAN_O2_test     : torch.Tensor
	STD_O2_test      : torch.Tensor
 
	LR_CO2_test      : torch.Tensor
	MEAN_CO2_test    : torch.Tensor
	STD_CO2_test     : torch.Tensor
 
	LR_H2O_test      : torch.Tensor
	MEAN_H2O_test    : torch.Tensor
	STD_H2O_test     : torch.Tensor
 
	LR_CO_test       : torch.Tensor
	MEAN_CO_test     : torch.Tensor
	STD_CO_test      : torch.Tensor
 
	LR_H2_test       : torch.Tensor
	MEAN_H2_test     : torch.Tensor
	STD_H2_test      : torch.Tensor
 
	LR_Unified_test  : torch.Tensor
	MEAN_Unified_test : torch.Tensor
	STD_Unified_test  : torch.Tensor	

@dataclass
class DataSplit_DualSensor_STD:
	U_train  		 : torch.Tensor
	Y_train  		 : torch.Tensor
	G_train  		 : torch.Tensor

	Gin_train_1  	 : torch.Tensor
	Yin_train_1   	 : torch.Tensor

	Gin_train_2  	 : torch.Tensor
	Yin_train_2   	 : torch.Tensor

	LR_T_train  	 : torch.Tensor
	MEAN_T_train     : torch.Tensor
	STD_T_train  	 : torch.Tensor

	LR_P_train  	 : torch.Tensor
	MEAN_P_train     : torch.Tensor
	STD_P_train  	 : torch.Tensor

	LR_Vx_train   	 : torch.Tensor
	MEAN_Vx_train 	 : torch.Tensor
	STD_Vx_train  	 : torch.Tensor

	LR_Vy_train      : torch.Tensor
	MEAN_Vy_train    : torch.Tensor
	STD_Vy_train     : torch.Tensor
   
	LR_O2_train      : torch.Tensor
	MEAN_O2_train    : torch.Tensor
	STD_O2_train     : torch.Tensor

	LR_CO2_train     : torch.Tensor
	MEAN_CO2_train   : torch.Tensor
	STD_CO2_train    : torch.Tensor

	LR_H2O_train     : torch.Tensor
	MEAN_H2O_train   : torch.Tensor
	STD_H2O_train    : torch.Tensor

	LR_CO_train      : torch.Tensor
	MEAN_CO_train    : torch.Tensor
	STD_CO_train     : torch.Tensor

	LR_H2_train      : torch.Tensor
	MEAN_H2_train    : torch.Tensor
	STD_H2_train     : torch.Tensor

	LR_Unified_train : torch.Tensor
	MEAN_Unified_train : torch.Tensor
	STD_Unified_train  : torch.Tensor

	U_test   		 : torch.Tensor
	Y_test  		 : torch.Tensor
	G_test   		 : torch.Tensor

	Gin_test_1  		 : torch.Tensor
	Yin_test_1   	 	 : torch.Tensor

	Gin_test_2  		 : torch.Tensor
	Yin_test_2   	 	 : torch.Tensor

	LR_T_test  	 	 : torch.Tensor
	MEAN_T_test      : torch.Tensor
	STD_T_test  	 : torch.Tensor

	LR_P_test  	 	 : torch.Tensor
	MEAN_P_test      : torch.Tensor
	STD_P_test  	 : torch.Tensor

	LR_Vx_test   	 : torch.Tensor
	MEAN_Vx_test 	 : torch.Tensor
	STD_Vx_test  	 : torch.Tensor

	LR_Vy_test       : torch.Tensor
	MEAN_Vy_test     : torch.Tensor
	STD_Vy_test      : torch.Tensor
    
	LR_O2_test       : torch.Tensor
	MEAN_O2_test     : torch.Tensor
	STD_O2_test      : torch.Tensor
 
	LR_CO2_test      : torch.Tensor
	MEAN_CO2_test    : torch.Tensor
	STD_CO2_test     : torch.Tensor
 
	LR_H2O_test      : torch.Tensor
	MEAN_H2O_test    : torch.Tensor
	STD_H2O_test     : torch.Tensor
 
	LR_CO_test       : torch.Tensor
	MEAN_CO_test     : torch.Tensor
	STD_CO_test      : torch.Tensor
 
	LR_H2_test       : torch.Tensor
	MEAN_H2_test     : torch.Tensor
	STD_H2_test      : torch.Tensor
 
	LR_Unified_test  : torch.Tensor
	MEAN_Unified_test : torch.Tensor
	STD_Unified_test  : torch.Tensor	