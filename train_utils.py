
import pathlib
import torch

def save_model(state_dict, path, epoch):
    torch.save(state_dict, path + str(epoch) + ".pth")