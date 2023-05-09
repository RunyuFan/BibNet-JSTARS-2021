import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
import torchvision
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from dataset import POI23DateSet
# from resnet_lulc import ResNet18, ResNet34, ResNet50, ResNet101
import argparse
# from ResNext import resnext50_32x4d
# from MSDnet import MSDNet
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
from Multimodel_train import MultiModalNet, FCViewer
from POI_train import POINet
from RSimg_train import RSNet
# sklearn.metrics.cohen_kappa_score(y1, y2, labels=None, weights=None, sample_weight=None)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rcParams["font.family"] = "Times New Roman"

def plot_confusion_matrix(cm,labels, title='Confusion Matrix of ResNet50'):
    # font1 = {'family': 'Times New Roman',
    #          'size':50}
    # font2 = {'family': 'Times New Roman',
    #          'size':35}
    # font3 = {'family': 'Times New Roman'}
    plt.imshow(cm)   #  , interpolation='nearest', cmap=plt.cm.binary
    # plt.title(title,fontsize=80) # ,fontfamily='Times New Roman')
    # plt.colorbar().ax.tick_params(labelsize=50)
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=45,fontsize=80) # ) # ,fontfamily='Times New Roman')
    plt.yticks(xlocations, labels,fontsize=80) # ,fontfamily='Times New Roman')
    plt.ylabel('True label',fontsize=80) # ,fontfamily='Times New Roman')
    plt.xlabel('Predicted label',fontsize=80) # ,fontfamily='Times New Roman')



def draw(y_true,y_pred,labels):
    tick_marks = np.array(range(len(labels))) + 0.5
    cm = confusion_matrix(y_true, y_pred)
    # np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(60, 60), dpi=120)

    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.3f" % (c,), color='red', fontsize=150, va='center', ha='center') # ) # ,fontfamily='Times New Roman') 50
        else:
            plt.text(x_val, y_val, 0, color='red', fontsize=150, va='center', ha='center') # ) # ,fontfamily='Times New Roman') 50
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15) # 0.15

    plot_confusion_matrix(cm_normalized, labels, title='Confusion matrix using POI data')
    # show confusion matrix
    plt.savefig('Confusion_POI.png', format='png')
    plt.close()

def main(args):
    # Create model
    # if not os.path.exists(args.model_path):
    #     os.makedirs(args.model_path)

    train_datasets = POI23DateSet('.\\GenShenzhenUFZ-8-2-4class\\train_data\\', './data/trainGenShenzhenUFZ-8-2-4class.txt', transforms=None)
    test_datasets = POI23DateSet('.\\GenShenzhenUFZ-8-2-4class\\test_data\\', './data/testGenShenzhenUFZ-8-2-4class.txt', transforms=None)

    train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                              batch_size=args.batch_size,
                                              shuffle=True)

    print("Train numbers:{:d}".format(len(train_datasets)))
    print("Test numbers:{:d}".format(len(test_datasets)))


    # model2 = torch.load('.\\model-UFZ\\UFZ-teacher-resnext50_32x4d_4class.pth')
    model2 = torch.load('.\\model-UFZ\\POI-UFZ-teacher-resnext50_32x4d_4class.pth')

    # model3 = torch.load('./model/MSDNet-shenzhen-5-5-lulc-6.pth')
    # print('model1 parameters:', sum(p.numel() for p in model1.parameters() if p.requires_grad))
    print('model2 parameters:', sum(p.numel() for p in model2.parameters() if p.requires_grad))
    # print('model3 parameters:', sum(p.numel() for p in model3.parameters() if p.requires_grad))
    # print(model1, model2)
    # cost
    # model1 = model1.to(device)
    model2 = model2.to(device)
    # model3 = model3.to(device)
    print('start eval')

    best_acc_1 = 0.
    best_acc_2 = 0.
    best_acc_3 = 0.

    # model1.eval()
    model2.eval()
    # model3.eval()
    y = []
    y_pred = []
    classes = ('Residential', 'Public service', 'Commercial', 'Industrial')
    # classes = ('airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud', 'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway', 'golf_course', 'ground_track_field', 'harbor', 'industrial_area', 'intersection', 'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park', 'mountain', 'overpass', 'palace', 'parking_lot', 'railway', 'railway_station', 'rectangular_farmland', 'river', 'roundabout', 'runway', 'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium', 'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station', 'wetland')
    class_correct1 = list(0. for i in range(args.num_class))
    class_total1 = list(0. for i in range(args.num_class))
    class_correct2 = list(0. for i in range(args.num_class))
    class_total2 = list(0. for i in range(args.num_class))
    class_correct3 = list(0. for i in range(args.num_class))
    class_total3 = list(0. for i in range(args.num_class))
    class_correct_all = list(0. for i in range(args.num_class))
    class_total_all = list(0. for i in range(args.num_class))
    correct_prediction_1 = 0.
    total_1 = 0
    correct_prediction_2 = 0.
    total_2 = 0
    correct_prediction_3 = 0.
    total_3 = 0
    correct_prediction_all = 0.
    total_all = 0
    with torch.no_grad():
        for images, labels, poi_dense_h, poi_dense_w in test_loader:
            # to GPU
            images = images.to(device)
            labels = labels.to(device)
            poi_dense_h = poi_dense_h.to(device)
            poi_dense_w = poi_dense_w.to(device)
            # poi_dense_h = torch.tensor(poi_dense_h, dtype=torch.float32)
            # poi_dense_w = torch.tensor(poi_dense_w, dtype=torch.float32)
            poi_dense_h = poi_dense_h.clone().detach().float()
            poi_dense_w = poi_dense_w.clone().detach().float()

            outputs2 = model2(poi_dense_h, poi_dense_w)

            y.append(labels)

            # outputs1 = model1(images)
            # _1, predicted1 = torch.max(outputs1, 1)
            # # y_pred.append(predicted1)
            # c1 = (predicted1 == labels).squeeze()
            # for label_idx in range(len(labels)):
            #     label = labels[label_idx]
            #     class_correct1[label] += c1[label_idx].item()
            #     class_total1[label] += 1
            # total_1 += labels.size(0)
            # # add correct
            # correct_prediction_1 += (predicted1 == labels).sum().item()


            # outputs2 = model2(images, poi_dense_h, poi_dense_w)
            _2, predicted2 = torch.max(outputs2, 1)
            y_pred.append(predicted2)
            c2 = (predicted2 == labels).squeeze()
            for label_idx in range(len(labels)):
                label = labels[label_idx]
                class_correct2[label] += c2[label_idx].item()
                class_total2[label] += 1
            total_2 += labels.size(0)
            # add correct
            correct_prediction_2 += (predicted2 == labels).sum().item()

            # outputs3 = model3(images)
            # _3, predicted3 = torch.max(outputs3, 1)
            # y_pred.append(predicted3)
            # c3 = (predicted3 == labels).squeeze()
            # for label_idx in range(len(labels)):
            #     label = labels[label_idx]
            #     class_correct3[label] += c3[label_idx].item()
            #     class_total3[label] += 1
            #
            # total_3 += labels.size(0)
            # # add correct
            # correct_prediction_3 += (predicted3 == labels).sum().item()
            #
            # blending_y_pred = outputs1 * 0.33 + outputs2 * 0.34 + outputs3 * 0.33
            # _, predicted_blending = torch.max(blending_y_pred, 1)
            # # y_pred.append(predicted_blending)
            # c_all = (predicted_blending == labels).squeeze()
            # for label_idx in range(len(labels)):
            #     label = labels[label_idx]
            #     class_correct_all[label] += c_all[label_idx].item()
            #     class_total_all[label] += 1
            #
            # total_all += labels.size(0)
            # # add correct
            # correct_prediction_all += (predicted_blending == labels).sum().item()
        t_l=torch.cat(y,dim=0)
        p_l=torch.cat(y_pred,dim=0)
        t_l=t_l.cpu().numpy()
        p_l = p_l.cpu().numpy()
    # for i in range(args.num_class):
    #     print('Model ResNet50 - Accuracy of %5s : %2f%%: Correct Num: %d in Total Num: %d' % (
    #         classes[i], 100 * class_correct1[i] / class_total1[i], class_correct1[i], class_total1[i]))
    # acc_1 = correct_prediction_1 / total_1
    # print("Total Acc Model ResNet50: %.4f" % (correct_prediction_1 / total_1))
    print('----------------------------------------------------')
    for i in range(args.num_class):
        print('Model - Accuracy of %5s : %2f%%: Correct Num: %d in Total Num: %d' % (
            classes[i], 100 * class_correct2[i] / class_total2[i], class_correct2[i], class_total2[i]))
    acc_2 = correct_prediction_2 / total_2
    print("Total Acc Model: %.4f" % (correct_prediction_2 / total_2))
    print('----------------------------------------------------')
    # for i in range(args.num_class):
    #     print('Model MSDNet - Accuracy of %5s : %2f%%: Correct Num: %d in Total Num: %d' % (
    #         classes[i], 100 * class_correct3[i] / class_total3[i], class_correct3[i], class_total3[i]))
    # print("Total Acc Model MSDNet: %.4f" % (correct_prediction_3 / total_3))
    # acc_3 = correct_prediction_3 / total_3
    # print('----------------------------------------------------')
    # for i in range(args.num_class):
    #     print('Model blending - Accuracy of %5s : %2f%%: Correct Num: %d in Total Num: %d' % (
    #         classes[i], 100 * class_correct_all[i] / class_total_all[i], class_correct_all[i], class_total_all[i]))
    # print("Total Acc Model blending: %.4f" % (correct_prediction_all / total_all))
    # print('####################################################')
    print(t_l, p_l)
    draw(t_l, p_l, classes)
    print(cohen_kappa_score(t_l, p_l))
    # correct_prediction = 0.
    # total = 0
    # for images, labels in test_loader:
    #     # to GPU
    #     images = images.to(device)
    #     labels = labels.to(device)
    #     # print prediction
    #     outputs = model(images)
    #     # equal prediction and acc
    #
    #     _, predicted = torch.max(outputs.data, 1)
    #     # val_loader total
    #     total += labels.size(0)
    #     # add correct
    #     correct_prediction += (predicted == labels).sum().item()

    # print("Acc: %.4f" % (correct_prediction / total))

# Save the model checkpoint
# if acc_1 > best_acc_1:
#     print('save new best acc_1', acc_1)
#     torch.save(model1, os.path.join(args.model_path, 'ResNet50-%s.pth' % (args.model_name)))
#     best_acc_1 = acc_1
# if acc_2 > best_acc_2:
#     print('save new best acc_2', acc_2)
#     torch.save(model2, os.path.join(args.model_path, 'resnext50_32x4d-%s.pth' % (args.model_name)))
#     best_acc_2 = acc_2
# if acc_3 > best_acc_3:
#     print('save new best acc_3', acc_3)
#     torch.save(model3, os.path.join(args.model_path, 'MSDNet-%s.pth' % (args.model_name)))
#     best_acc_3 = acc_3
# print("Model save to %s."%(os.path.join(args.model_path, 'model-%s.pth' % (args.model_name))))
# print('save new best acc_1', best_acc_1)
# print('save new best acc_2', best_acc_2)
# print('save new best acc_3', best_acc_3)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train hyper-parameter')
    parser.add_argument("--num_class", default=4, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    # parser.add_argument("--net", default='ResNet50', type=str)
    # parser.add_argument("--depth", default=50, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    # parser.add_argument("--num_workers", default=2, type=int)
    # parser.add_argument("--model_name", default='lulc-6-fintune-GID', type=str)
    # parser.add_argument("--model_path", default='./model', type=str)
    # parser.add_argument("--pretrained", default=False, type=bool)
    # parser.add_argument("--pretrained_model", default='./model/ResNet50.pth', type=str)
    args = parser.parse_args()

    main(args)
