from utils import *
import torch
import time
import config.efficient_net7 as conf

def model_train(model, criterion, optimizer, trainloader, testloader, use_gpu, epoch, scheduler, save_model_path):
    model.train()
    losses = AverageMeter() # loss의 평균을 구하는 함수
    start = time.time()
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
    
        if (iter+1) % conf.print_freq == 0: #매 print_freq iteration마다 아래를 출력
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})\t lr {}\t time {:.3f}" \
                    .format(iter+1, len(trainloader), losses.val, losses.avg, lr, time.time()-start))
            start = time.time()

        if (iter+1) % 30 == 0 :
            save_model_name = save_model_path + '_' + str(epoch) + '_' + str(iter+1) + '.pth'
            torch.save(model.module.state_dict(), save_model_name)
            # print('save :', save_model_name)
    
    # 매 stepsize마다 learning rate를 0.1씩 감소하는 scheduler 실행.
    if conf.is_scheduler == True: 
        scheduler.step()

    # Accuracy를 출력
    print("==> Test Accuracy")
    acc, err = model_test(model, testloader, use_gpu)
    print("Accuracy (%): {}\t Error rate(%): {}".format(acc, err))

def model_test(model, testloader, use_gpu):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad(): #parameter을 갱신하지 않음(backpropagation을 하지않음)
        for i, (data, labels, image_name) in enumerate(testloader):
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