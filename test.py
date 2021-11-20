import torch
import csv
import time
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def test(testloader, model, classes, load_model_path, out_path, multi_test_size, device):
    # model load
    if device == "cuda":
        model.load_state_dict(torch.load(load_model_path))
    else : 
        model.load_state_dict(torch.load(load_model_path, map_location=torch.device(device)))

    model.eval()

    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    pred = []
    y_true = []
    y_pred = []
    outs = []
    multi_test_len = len(multi_test_size)

    start = time.time()
    with torch.no_grad():
        for i, (data, labels, image_name) in enumerate(testloader):
            if device == "cuda":
                data = data.cuda()
            outputs = model(data)

            outs.append(outputs)

            if (i+1) % len(multi_test_size) == 0 :
                tmp = 0
                for k in range(multi_test_len):
                    tmp = tmp + outs[k]
                outs = []

                _, predicted = torch.max(tmp.data, 1)

                y_pred.append(predicted.cpu().tolist())
                y_true.append(labels.tolist())

                total += labels.size(0)
                correct += (predicted.cpu() == labels).sum().item()
                pred.append([image_name[0], predicted.cpu().item()])

                for label, prediction in zip(labels, predicted.cpu()):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

    print("inference time :", time.time() - start)
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
    
    # confusion matrix
    cf = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix")
    print(cf)

    # prediction csv파일로 저장
    f = open(out_path, 'w', newline='')
    wr = csv.writer(f)
    wr.writerows(pred)
    f.close()
    print(out_path, 'file saved!!')
