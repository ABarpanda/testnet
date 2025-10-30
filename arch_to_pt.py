import json
# import torch
# import torch.nn as nn
from typing import List

def write_pytorch(architecture_json: List, save_path: str = "generated_model.py"):

    assert architecture_json[0]["kind"] == "Input", "First layer must be Input layer"
    import layer_functions as lf

    functions = {
        "Conv2d": "nn.Conv2d",
        "MaxPool2d": "nn.MaxPool2d",
        "Dropout": "nn.Dropout",
        "Flatten": "nn.Flatten",
        "Linear": "nn.Linear",
        "ReLU": "nn.ReLU",
        "BatchNorm2d": "nn.BatchNorm2d",
        "Sigmoid": "nn.Sigmoid",
        "Tanh": "nn.Tanh",
        "AvgPool2d": "nn.AvgPool2d",
        "Softmax": "nn.Softmax",
        "BatchNorm2d": "nn.BatchNorm2d",
    }

    layers = ""
    for layer in architecture_json[1:]:
        kind = layer["kind"]
        if kind in functions:
            func = getattr(lf, kind.lower())
            if kind == "Conv2d":
                layer_code = func(layer, architecture_json)  
            else:
                layer_code = func(layer)
            # print(f'Generated code for layer {layer}: {layer_code}')
            layers += f"            {layer_code},\n"
        else:
            raise ValueError(f"Unsupported layer type: {kind}")

    code = f"""
import torch
import torch.nn as nn

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.model = nn.Sequential(
{layers}    )

    def forward(self, x):
        return self.model(x)
        """
    return code


if __name__ == "__main__":
    with open("architecture.json", "r") as file:
        architecture_json = json.load(file)

    code = write_pytorch(architecture_json)
    print("\nGenerated PyTorch Code:\n")
    print(code)
