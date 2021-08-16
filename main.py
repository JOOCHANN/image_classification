import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
from pathlib import Path
import torch
import os
import csv
from sklearn.metrics import precision_recall_fscore_support

from model.efficientnet.model import EfficientNet
from torch.utils.data import DataLoader
from utils import *
from dataset import SMOKE

classes = ['working_factory', 'unworking_factory', 'cloud']

seed = '1'
lr = 0.01
max_epoch = 10
print_freq = 10
model_name = 'efficientnet-b7'
num_classes = len(classes)
weight_decay = 5e-04
momentum = 0.9
batch = 16
stepsize = 7
is_scheduler = True
gamma = 0.1
data_path = '/data/data_server/pjc/planet_data_classification'
save_model_path = './work_dirs/efficientnet-b7/'
load_model_path = './work_dirs/efficientnet-b7/_9.pth'
out_path = '.predictions.csv'

def main(pause):
    # SEED
    torch.manual_seed(seed)
    use_gpu = torch.cuda.is_available()
    print("Use gpu :", use_gpu)
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(seed)

    # data loading
    if pause == 0:
        print("Train dataset loading")
        train_path = os.path.join(data_path, 'train')
        smoke_trainset = SMOKE(classes, train_path, isTrain = True)
        trainloader = DataLoader(dataset=smoke_trainset, batch_size=batch, shuffle=True)
    elif pause == 1:
        print("Test dataset loading")
        test_path = os.path.join(data_path, 'test_v3')
        smoke_trainset = SMOKE(classes, test_path, isTrain = False)
        testloader = DataLoader(dataset=smoke_trainset, batch_size=1, shuffle=False)
    
    print("number of data :", len(smoke_trainset), "End dataset loading")

    # model loading
    print("Creating model: {}".format(model_name))
    model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    # print('Model input size', EfficientNet.get_image_size(model_name))
    # print("End Creating model: {}".format(model_name))
    if pause == 1 :
        model.to("cuda")
        bind_model(testloader, model)

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
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

    #learning rate를 일정 epoch마다 감소
    if is_scheduler == True:
        # 매 stepsize마다 learning rate를 0.1씩 감소하는 scheduler 생성
        scheduler = lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gamma)

        # CyclicLR
        # scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-2, step_size_up=2000)

        # warmup
        # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, 
        # total_epoch=1, after_scheduler=scheduler)

    for epoch in range(max_epoch):
        epoch = epoch
        print("==> Epoch {}/{}".format(epoch+1, max_epoch))

        #model train
        train(model, criterion, optimizer, trainloader, use_gpu, epoch)

        # 매 stepsize마다 learning rate를 0.1씩 감소하는 scheduler 실행.
        if is_scheduler == True:
            if epoch == 0:
                print("epoch 0... no learning rate step")
            else : 
                scheduler.step()

        #Train Accuracy를 출력
        print("==> Train")
        acc, err = test(model, trainloader, use_gpu)
        print("Accuracy (%): {}\t Error rate(%): {}".format(acc, err))

        #save model
        save_model_name = save_model_path + '_' + str(epoch) + '.pth'
        torch.save(model.module.state_dict(), save_model_name)
        # save_model(model, 'e{}'.format(epoch))

def train(model, criterion, optimizer, trainloader, use_gpu, epoch):
    model.train()
    losses = AverageMeter() # loss의 평균을 구하는 함수
    for iter, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()

        outputs = model(data) # 각 클래스에대한 softmax값이 output으로 나옴, 개수는 batchsize만큼
        loss = criterion(outputs, labels) # 정답 label과 output과 비교하여 loss 측정
        optimizer.zero_grad() # optimizer 초기화
        loss.backward() # loss backpropagation
        optimizer.step() # parameter update
        # AverageMeter function에 update함수 적용 -> loss의 평균을 losses에 저장
        losses.update(loss.item(), labels.size(0)) # labels.size(0) = batch size
    
        if (iter+1) % print_freq == 0: #매 print_freq iteration마다 아래를 출력
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})\t lr {}" \
                    .format(iter+1, len(trainloader), losses.val, losses.avg, lr))

        if (iter+1) % 30 == 0 :
            save_model_name = save_model_path + '_' + str(epoch) + '_' + str(iter+1) + '.pth'
            torch.save(model.module.state_dict(), save_model_name)
            # print('save :', save_model_name)

def test(model, testloader, use_gpu):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad(): #parameter을 갱신하지 않음(backpropagation을 하지않음)
        for i, (data, labels) in enumerate(testloader):
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            outputs = model(data)
            # Softmax를 사용했으므로 outputs는 확률값임 우리는 이 확률중 제일 높은 하나의 index를 뽑아야함
            predictions = outputs.data.max(1)[1]
            # labels.size(0) = batch size이므로 total은 현재 본 데이터의 총 개수
            total += labels.size(0)
            # 예측한 predictions의 index값과 labels의 값과 같다면 correct를 count함
            # predictions와 labels는 벡터이므로 .sum()을 이용하여 count가능
            correct += (predictions == labels.data).sum()

    # acc = 정답을 맞춘 개수 / 전체 데이터
    acc = correct.item() * 1.0 / total
    # err = 1 - acc
    err = 1.0 - acc
    return acc, err

def bind_model(testloader, model):
    # model load
    model.load_state_dict(torch.load(load_model_path))
    model.eval()
    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    pred = []
    y_true = []
    y_pred = []

    with torch.no_grad():
        for i, (data, labels, image_name) in enumerate(testloader):
            data = data.cuda()
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)

            y_pred.append(predicted.cpu().tolist())
            y_true.append(labels.tolist())

            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()
            pred.append([image_name[0], predicted.cpu().item()])

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average=None)

    # precision, recall, f1score 출력
    print('precision: \t{}'.format(precision))
    print('recall: \t{}'.format(recall))
    print('fscore: \t{}'.format(fscore))
    print('support: \t{}'.format(support))

    # 결과 출력
    print('Accuracy of the all test images: %d %%' % (100 * correct / total))
    # for classname, correct_count in correct_pred.items():
    #     accuracy = 100 * float(correct_count) / total_pred[classname]
    #     print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
    
    # prediction csv파일로 저장
    f = open(out_path, 'w', newline='')
    wr = csv.writer(f)
    wr.writerows(pred)
    f.close()
    print(out_path[1:], 'file saved!!')

    exit(1)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--mode', default='train')
    args = parser.parse_args()

    mkdir_if_not_exists(save_model_path)

    if args.mode == 'test':
        print("Testing!!!")
        main(pause = 1)
    
    if args.mode == 'train':
        print("Training!!!")
        main(pause = 0)