import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import argparse
import torch
import os
from model.efficientnet.model import EfficientNet
from torch.utils.data import DataLoader
from utils import *
from dataset import SMOKE_DEMO
import config.efficient_net7 as conf
from train import model_train
from test import *

classes = ['working_factory', 'unworking_factory', 'cloud']

def main():
    # SEED
    torch.manual_seed(conf.seed)
    use_gpu = torch.cuda.is_available()
    print("Use gpu :", use_gpu)
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(conf.seed)

    # data loading
    print("Demo dataset loading")
    demo_path = os.path.join(data_path)
    smoke_demoset = SMOKE_DEMO(classes, demo_path)
    testloader = DataLoader(dataset=smoke_demoset, batch_size=1, shuffle=False)

    # model loading
    print("Creating model: {}".format(conf.model_name))
    model = EfficientNet.from_pretrained(conf.model_name, num_classes=conf.num_classes)
    model.to("cuda")
    model.load_state_dict(torch.load(load_model_path))

    pred = []

    with torch.no_grad():
        for i, (data, image_name) in enumerate(testloader):
            data = data.cuda()
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            pred.append([image_name[0], classes[predicted.cpu().item()]])

    f = open(out_file, 'w', newline='')
    wr = csv.writer(f)
    wr.writerows(pred)
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='smoke classification')
    parser.add_argument('--load_model_path', default='./work_dirs/efficientnet-b7/best_epoch.pth', 
                            help='Path of model weights to be loaded')
    parser.add_argument('--out_file', default='predictions.csv', 
                            help='csv file to save the result')
    parser.add_argument('--demo_data_path', default='./demo_img', 
                            help='Data path')
    args = parser.parse_args()

    load_model_path = args.load_model_path
    out_file = args.out_file
    data_path = args.demo_data_path

    main()