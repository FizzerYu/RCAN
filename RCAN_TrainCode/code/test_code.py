import torch

import utility
import data
import model
import loss
from option import args    # 参数设置
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)   # dataset&dataloader
    print(loader)
    for index,data in enumerate(loader.loader_test):  #loader_test
        print(index,len(data),data[0].shape,data[0].shape,data[2])
