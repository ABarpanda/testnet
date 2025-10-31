import torch
import torch.nn as nn
import torch.optim as optim
import json
from arch_to_pt import write_pytorch
from modify_arch import modify_json

with open("architecture5.json", "r") as file:
    architecture_json = json.load(file)

num_of_images = 100

architecture_json = modify_json(architecture_json,num_of_images)

model_code = write_pytorch(architecture_json)
exec(model_code)

model = CustomNet()
print(model)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

input_shape = [int(s) for s in str(architecture_json[0]["params"]["shape"]).split(",")]
x = torch.randn(tuple([num_of_images] + input_shape)).to(device)
y = torch.randn(num_of_images, int(architecture_json[-1]["params"].get("out_features"))).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def hook_fn(module, input, output):
    print(f"Layer: {module.__class__.__name__}, Output Shape: {output.shape}")

epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # for name, module in model.named_modules():
    #     if not list(module.children()):  # Only hook leaf modules
    #         # hooks.append(module.register_forward_hook(hook_fn))
    #         print(module.register_forward_hook(hook_fn))

    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")