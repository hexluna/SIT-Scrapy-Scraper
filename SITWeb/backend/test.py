import torch
print(torch.cuda.is_available())  # Should be True
print(torch.version.cuda)         # Should say '12.1'
print(torch.cuda.get_device_name(0))