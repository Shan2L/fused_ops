import torch
import concatcpp
import numpy as np
from torch.profiler import profile, ProfilerActivity

with profile(
	activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
	on_trace_ready=torch.profiler.tensorboard_trace_handler("./torch_cat_embedding"),
	profile_memory=True,
	with_stack=True,
	record_shapes=True,
    use_cuda=True) as prof:
    tensor_list = []
    for i in range(48):
        a = torch.randint(low=0, high=10, size=(1, 40, 512, 1024)).int()
        tensor_list.append(a)z

    weight = torch.randn((11, 10), dtype=torch.float)
    golden_cat = torch.cat(tensor_list, dim=0)
    golden_re = torch.nn.functional.embedding(golden_cat, weight);

    
# with open('golden.txt', 'w') as gf, open("cus.txt", 'w') as cf:
#     for item in golden_result.storage():
#         gf.write(str(item))
#         gf.write('\n')

#     for item in cus_result.storage():
#         cf.write(str(item))
#         cf.write('\n')