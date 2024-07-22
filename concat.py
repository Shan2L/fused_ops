import torch
import concatcpp


a = torch.randn(6, 3, 7).float()
b = torch.randn(6, 4, 7).float()
c = torch.randn(6, 5, 7).float()

tensors = [a, b,c ]

cus_result = concatcpp.forward(tensors, 1);

golden_reuslt = torch.cat([a, b, c], dim=1);

print(cus_result == golden_reuslt)