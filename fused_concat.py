from torch.profiler import profile, ProfilerActivity
import torch
import concatcpp

with profile(
	activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
	on_trace_ready=torch.profiler.tensorboard_trace_handler("./fused_concat_embedding"),
	profile_memory=True,
	with_stack=True,
	record_shapes=True,
    use_cuda=True) as prof:
    tensor_list = []
    for i in range(48):
        a = torch.randint(low=0, high=10, size=(1, 40, 512, 1024)).int()
        tensor_list.append(a)

    weight = torch.randn((11, 10), dtype=torch.float)

    cus_result = concatcpp.fused_embedding(tensor_list, weight, 0);
