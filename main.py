import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import argparse
import torch
import os
from model.efficientnet.model import EfficientNet
from torch.utils.data import DataLoader
from utils import *
from dataset import SMOKE
import config.efficient_net7 as conf
from train import model_train
from test import *

classes = ['working_factory', 'unworking_factory', 'cloud']

def main(mode):
    # SEED
    torch.manual_seed(conf.seed)
    use_gpu = torch.cuda.is_available()
    print("Use gpu :", use_gpu)
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(conf.seed)

    # data loading
    print("Train dataset loading")
    train_path = os.path.join(data_path, 'train')
    smoke_trainset = SMOKE(classes, train_path, isTrain=True, multi_size=conf.train_size)
    trainloader = DataLoader(dataset=smoke_trainset, batch_size=conf.batch, shuffle=True)

    print("Test dataset loading")
    test_path = os.path.join(data_path, 'test_v3')
    smoke_trainset = SMOKE(classes, test_path, isTrain=False, multi_size=conf.multi_test_size) # multi size testing
    testloader = DataLoader(dataset=smoke_trainset, batch_size=1, shuffle=False)

    # model loading
    print("Creating model: {}".format(conf.model_name))
    model = EfficientNet.from_pretrained(conf.model_name, num_classes=conf.num_classes)
    # print('Model input size', EfficientNet.get_image_size(model_name))
    if mode == 'test' :
        model.to("cuda")
        test(testloader, model, classes, load_model_path, out_path, conf.multi_test_size)
        exit(1)

    # model to GPU
    if torch.cuda.device_count() > 1 :
        print("Use", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model).cuda()
    else :
        model.to("cuda")
    print("Loaded")

    #Loss Function
    criterion = nn.CrossEntropyLoss()

    #Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay, momentum=conf.momentum)

    #learning rate scheduler
    if conf.is_scheduler == True:
        # 매 stepsize마다 learning rate를 0.1씩 감소하는 scheduler 생성
        scheduler = lr_scheduler.StepLR(optimizer, step_size=conf.stepsize, gamma=conf.gamma)

    # train model
    for epoch in range(conf.max_epoch):
        epoch = epoch
        print("==> Epoch {}/{}".format(epoch+1, conf.max_epoch))

        # train model
        model_train(model, criterion, optimizer, trainloader, testloader, use_gpu, epoch, scheduler, save_model_path)

        # save model
        save_model_name = save_model_path + '_' + str(epoch+1) + '.pth'
        torch.save(model.module.state_dict(), save_model_name)
        # save_model(model, 'e{}'.format(epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='smoke classification')
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--load_model_path', default='./work_dirs/efficientnet-b7/randaug_2_14_best_epoch/_9.pth', 
                            help='Path of model weights to be loaded')
    parser.add_argument('--out_path', default='predictions.csv', 
                            help='csv file to save the result')
    parser.add_argument('--save_model_path', default='./work_dirs/efficientnet-b7/', 
                            help='Path to store model weights')
    parser.add_argument('--data_path', default='/home/ubuntu/data/smoke_classification', 
                            help='Data path')
    args = parser.parse_args()

    mode = args.mode
    load_model_path = args.load_model_path
    out_path = args.out_path
    save_model_path = args.save_model_path
    data_path = args.data_path

    if mode == 'test':
        print("Testing!!!")
    
    if mode == 'train':
        mkdir_if_not_exists(save_model_path)
        print("Training!!!")
    
    main(mode)