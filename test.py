import torch
import csv
from sklearn.metrics import precision_recall_fscore_support

def test(testloader, model, classes, load_model_path, out_path):
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