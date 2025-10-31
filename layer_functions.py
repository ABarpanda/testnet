def conv2d(layer, architecture_json: dict):
    prev_layer_id = layer["inputs"][0]
    prev_layer = next(l for l in architecture_json if l["id"] == prev_layer_id)
    in_ch = prev_layer["params"].get("out_channels")
    if in_ch is None:
        shape = prev_layer["params"].get("shape")
        if shape:
            in_ch = int(shape.split(",")[0])
        else:
            in_ch = 3
    out_ch = layer["params"].get("out_channels", 16)
    k = layer["params"].get("kernel_size", 3)
    s = layer["params"].get("stride", 1)
    p = layer["params"].get("padding", 1)
    return f"nn.Conv2d(in_channels = {in_ch}, out_channels = {out_ch}, kernel_size={k}, stride={s}, padding={p})"

def batchnorm2d(layer):
    num_features = layer["params"].get("num_features", 16)
    return f"nn.BatchNorm2d(num_features={num_features})"

def maxpool2d(layer):
    k = layer["params"].get("kernel_size", 2)
    s = layer["params"].get("stride", 2)
    return f"nn.MaxPool2d(kernel_size={k}, stride={s})"

def avgpool2d(layer):
    k = layer["params"].get("kernel_size", 2)
    s = layer["params"].get("stride", 2)
    return f"nn.AvgPool2d(kernel_size={k}, stride={s})"

def dropout(layer):
    p = layer["params"].get("p", 0.5)
    return f"nn.Dropout(p={p})"

def flatten(_):
    return "nn.Flatten()"

def linear(layer):
    _, inp = layer["params"].get("in_shape", 128)
    out = layer["params"].get("out_features", 10)
    return f"nn.Linear({inp}, {out})"

def relu(_):
    return "nn.ReLU()"

def softmax(layer):
    dim = layer["params"].get("dim", 1)
    return f"nn.Softmax(dim={dim})"

if __name__ == "__main__":
    import json
    with open("architecture.json", "r") as file:
        architecture_json = json.load(file)