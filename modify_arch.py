import json
import math
from typing import Any
import layer_functions

with open("architecture.json", "r") as file:
    architecture_json = json.load(file)

def modify_input_layer(layer:dict, no_of_samples:int)->tuple:
    assert layer["kind"]=="Input", "The first layer must be Input layer"
    params = dict(layer["params"])
    params["out_shape"] = [no_of_samples] + [int(i) for i in str(params["shape"]).split(",")]
    return tuple(params["out_shape"])

def modify_json(architecture_json:list, no_of_samples:int)->list:
    for idx in range(len(architecture_json)):
        layer = dict(architecture_json[idx])
        if idx == 0:
            assert layer["kind"]=="Input", "The first layer must be Input layer"
            layer["params"]["out_shape"] = modify_input_layer(layer, no_of_samples)
            # print(layer["params"]["out_shape"])
            # print(layer)
        else:
            prev_layer = next(l for l in architecture_json if l["id"] == layer["inputs"][0])
            layer["params"]["in_shape"] = prev_layer["params"]["out_shape"]

            if layer["kind"] == "Conv2d":
                n,cin,hin,win = layer["params"]["in_shape"]
                cout = layer["params"]["out_channels"]
                # hout = math.floor((hin + 2*layer["params"]["padding"] - layer["params"]["dilation"]*(layer["params"]["kernel_size"]-1)-1)/(layer["params"]["stride"]))+1
                hout = math.floor((hin + 2*layer["params"]["padding"] - 1*(layer["params"]["kernel_size"]-1)-1)/(layer["params"]["stride"]))+1
                # wout = math.floor((win + 2*layer["params"]["padding"] - layer["params"]["dilation"]*(layer["params"]["kernel_size"]-1)-1)/(layer["params"]["stride"]))+1
                wout = math.floor((win + 2*layer["params"]["padding"] - 1*(layer["params"]["kernel_size"]-1)-1)/(layer["params"]["stride"]))+1
                layer["params"]["out_shape"] = tuple([n,cout,hout,wout])
            elif layer["kind"] == "Flatten":
                in_shape = layer["params"]["in_shape"]
                if len(in_shape) == 4:
                    n, c, h, w = in_shape
                    layer["params"]["out_shape"] = (n, c * h * w)
                elif len(in_shape) == 3:
                    c, h, w = in_shape
                    layer["params"]["out_shape"] = (c * h * w,)
                elif len(in_shape) == 2:
                    # already flattened
                    layer["params"]["out_shape"] = in_shape
            else:
                layer["params"]["out_shape"] = layer["params"]["in_shape"]
            # print(layer)
    return architecture_json