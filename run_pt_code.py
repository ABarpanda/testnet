import torch
import torch.nn as nn
import torch.optim as optim
import json
from arch_to_pt import write_pytorch

with open("architecture5.json", "r") as file:
    architecture_json = json.load(file)

model_code = write_pytorch(architecture_json)
exec(model_code)

model = CustomNet()
print(model)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

x = torch.randn(8, 3, 224, 224).to(device)
y = torch.randn(8, 802816).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 3
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")