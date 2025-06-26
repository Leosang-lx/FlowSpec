import torch

temp = torch.tensor(2., dtype=torch.float16, device='cuda:1') # Pytorch will get 2MB reserved memory from vram
print(torch.cuda.list_gpu_processes(device='cuda:1'))
# print(torch.cuda.memory_summary(device='cuda:1'))
print('memory_allocated:' + str(torch.cuda.memory_allocated(device='cuda:1') / (1024 * 1024)) + ' MB')
print('max_memory_allocated:' + str(torch.cuda.max_memory_allocated(device='cuda:1') / (1024 * 1024)) + ' MB')
print('memory_reserved:' + str(torch.cuda.memory_reserved(device='cuda:1') / (1024 * 1024)) + ' MB')
print('max_memory_reserved:' + str(torch.cuda.max_memory_reserved(device='cuda:1') / (1024 * 1024)) + ' MB')

del temp
torch.cuda.empty_cache()

print(torch.cuda.list_gpu_processes(device='cuda:1'))
# print(torch.cuda.memory_summary(device='cuda:1'))
print('memory_allocated:' + str(torch.cuda.memory_allocated(device='cuda:1') / (1024 * 1024)) + ' MB')
print('max_memory_allocated:' + str(torch.cuda.max_memory_allocated(device='cuda:1') / (1024 * 1024)) + ' MB')
print('memory_reserved:' + str(torch.cuda.memory_reserved(device='cuda:1') / (1024 * 1024)) + ' MB')
print('max_memory_reserved:' + str(torch.cuda.max_memory_reserved(device='cuda:1') / (1024 * 1024)) + ' MB')