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
    if mode =='train':
        print("Train dataset loading")
        smoke_trainset = SMOKE(classes, train_data_path, isTrain=True, multi_size=conf.train_size)
        trainloader = DataLoader(dataset=smoke_trainset, batch_size=conf.batch, shuffle=True)

    print("Test dataset loading")
    smoke_trainset = SMOKE(classes, test_data_path, isTrain=False, multi_size=conf.multi_test_size) # multi size testing
    testloader = DataLoader(dataset=smoke_trainset, batch_size=1, shuffle=False)

    # model loading
    print("Creating model: {}".format(conf.model_name))
    model = EfficientNet.from_pretrained(conf.model_name, num_classes=conf.num_classes)
    # print('Model input size', EfficientNet.get_image_size(model_name))
    
    if use_gpu == True:
        device = "cuda"
    else :
        device = "cpu"

    if mode == 'test' :
        model.to(device)
        test(testloader, model, classes, load_model_path, out_path, conf.multi_test_size, device)
        exit(1)

    # model to GPU
    if torch.cuda.device_count() > 1 :
        print("Use", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model).cuda()
    else :
        model.to(device)
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
    parser.add_argument('--mode', choices=['train', 'test'], default='test')
    parser.add_argument('--load_model_path', default='./work_dirs/efficientnet-b7/best_epoch.pth', 
                            help='Path of model weights to be loaded')
    parser.add_argument('--out_path', default='test_predictions.csv', 
                            help='csv file to save the result')
    parser.add_argument('--save_model_path', default='./work_dirs/efficientnet-b7/', 
                            help='Path to store model weights')
    parser.add_argument('--train_data_path', default='/data/data_server/pjc/smoke_classification/train_v3', 
                            help='Data path')
    parser.add_argument('--test_data_path', default='/data/data_server/pjc/smoke_classification/test_v2', # test_v2, final_test
                            help='Data path')
    args = parser.parse_args()

    mode = args.mode
    load_model_path = args.load_model_path
    out_path = args.out_path
    save_model_path = args.save_model_path
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path

    if mode == 'test':
        print("Testing!!!")
    
    if mode == 'train':
        mkdir_if_not_exists(save_model_path)
        print("Training!!!")
    
    main(mode)