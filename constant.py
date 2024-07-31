
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