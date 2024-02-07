import sys
sys.path.append("..")
import modules as dcnv3
import torch
import torch.nn as nn
import torch.optim as optim

# Define the model
core_op = getattr(dcnv3, 'DCNv3_pytorch')
dcn = core_op(
    channels=3,
    kernel_size=3,
    stride=1,
    pad=1,
    dilation=1,
    group=1,
    offset_scale=1.0,
    act_layer='GELU',
    norm_layer='LN',
    dw_kernel_size=None,  # for InternImage-H/G
    center_feature_scale=False
)

# Dummy input
dummy_input = torch.randn(1, 256, 256, 3)
# Forward pass
output = dcn(dummy_input)

tmp = [param.clone() for param in dcn.parameters()]

print("##################")
# Assuming you have some target tensor
target = torch.randn_like(output)

# Define a loss function (MSE for demonstration)
criterion = nn.MSELoss()

# Compute the loss
loss = criterion(output, target)
# Zero the gradients
dcn.zero_grad()

# Backward pass
loss.backward()

# Update the weights
optimizer = optim.SGD(dcn.parameters(), lr=0.1)
optimizer.step()

# Check if the parameters are the same
for param1, param2 in zip(tmp, dcn.parameters()):
    if not torch.equal(param1, param2):
        print("Parameters are not the same!")
        print("param1:", param1)
        print("param2:", param2)

  
