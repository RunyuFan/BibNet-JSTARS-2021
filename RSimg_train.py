import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
import torchvision
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from dataset import POI23DateSet
# from resnet_lulc_AID import ResNet18, ResNet34, ResNet50, ResNet101
import argparse
# from ResNext_AID import resnext50_32x4d
# from MSDnet_AID import msdnet
from torchvision.models import resnet50, resnext50_32x4d, densenet121
import pretrainedmodels
from pretrainedmodels.models import *
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FCViewer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class RSNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.img_model=DPN26()
        img_model = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet')
        # self.img_model.load_state_dict(torch.load(args.pretrained_model))
        # self.channel_in_2 = self.img_model.fc.in_features  # 获取fc层的输入通道数
        # # print(self.channel_in_2)
        # self.img_model.fc = nn.Linear(self.channel_in_2, 10)
        # self.img_model = nn.Sequential(*self.img_model)
        self.img_model = list(img_model.children())[:-2]
        self.img_model.append(nn.AdaptiveAvgPool2d(1))

        # self.img_encoder = list(img_model.children())[:-2]
        # self.img_encoder.append(nn.AdaptiveAvgPool2d(1))
        self.img_model = nn.Sequential(*self.img_model)

        self.img_fc = nn.Sequential(
            FCViewer(),
            nn.Linear(img_model.last_linear.in_features, args.num_class)
        )

    def forward(self, x_img, x_poi_1, x_poi_2):

        # 然后把resnet的fc层替换成自己分类类别的fc层
         #
        # self.img_model = torch.nn.Sequential(self.img_model)

        x_img = self.img_model(x_img)
        x_img = self.img_fc(x_img)
        # print(x_img.shape)

        # x_poi_1 = self.poi_model1(x_poi_1)
        # x_poi_2 = self.poi_model2(x_poi_2)
        # # print(x_poi_1.shape, x_poi_2.shape)
        # x_cat = torch.cat((x_poi_1,x_poi_2),1)
        # x_cat = torch.cat((x_img,x_cat),1)

        # print(x_cat.shape)
        # x_cat = self.cls_cat(x_img)
        return x_img

def main(args):
    # Create model
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    train_datasets = POI23DateSet('.\\GenShenzhenUFZ-8-2-4class\\train_data\\', './data/trainGenShenzhenUFZ-8-2-4class.txt', transforms=None)
    test_datasets = POI23DateSet('.\\GenShenzhenUFZ-8-2-4class\\test_data\\', './data/testGenShenzhenUFZ-8-2-4class.txt', transforms=None)
    # test_datasets = CarDateSet('G:\\LULC\\PytorchLULC\\AID-RESISC30-unlabel-test\\test_data\\', './data/testAID.txt', transforms=None)

    train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                              batch_size=args.batch_size,
                                              shuffle=True)

    print("Train numbers:{:d}".format(len(train_datasets)))
    print("Test numbers:{:d}".format(len(test_datasets)))

    model2 = RSNet()

    print('model2 parameters:', sum(p.numel() for p in model2.parameters() if p.requires_grad))
    # print('model3 parameters:', sum(p.numel() for p in model3.parameters() if p.requires_grad))

    # model1 = model1.to(device)
    model2 = model2.to(device)
    # model3 = model3.to(device)
    # cost1 = nn.CrossEntropyLoss().to(device)
    cost2 = nn.CrossEntropyLoss().to(device)
    # cost3 = nn.CrossEntropyLoss().to(device)
    # Optimization
    # optimizer1 = optim.Adam(model1.parameters(), lr=args.lr, weight_decay=1e-6)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.lr, weight_decay=1e-6)
    # optimizer3 = optim.Adam(model3.parameters(), lr=args.lr, weight_decay=1e-6)


    # best_acc_1 = 0.
    best_acc_2 = 0.
    best_epoch = 0
    # best_acc_3 = 0.

    for epoch in range(1, args.epochs + 1):
        # model1.train()
        model2.train()
        # model3.train()
        # start time
        start = time.time()
        index = 0
        for images, labels, poi_dense_h, poi_dense_w in train_loader:
            images = images.to(device)
            # print(images.shape)
            labels = labels.to(device)
            poi_dense_h = poi_dense_h.to(device)
            # print(images.shape)
            poi_dense_w = poi_dense_w.to(device)
            # poi_dense_h = torch.tensor(poi_dense_h, dtype=torch.float32)
            # poi_dense_w = torch.tensor(poi_dense_w, dtype=torch.float32)
            poi_dense_h = poi_dense_h.clone().detach().float()
            poi_dense_w = poi_dense_w.clone().detach().float()

            # Forward pass
            # outputs1 = model1(images)
            outputs2 = model2(images, poi_dense_h, poi_dense_w)
            # outputs3 = model3(images)
            # loss1 = cost1(outputs1, labels)
            loss2 = cost2(outputs2, labels)
            # loss3 = cost3(outputs3, labels)

            # if index % 10 == 0:
                # print (loss)
            # Backward and optimize
            # optimizer1.zero_grad()
            optimizer2.zero_grad()
            # optimizer3.zero_grad()
            # loss1.backward()
            loss2.backward()
            # loss3.backward()
            # optimizer1.step()
            optimizer2.step()
            # optimizer3.step()
            index += 1


        if epoch % 1 == 0:
            end = time.time()
            # print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!" % (epoch, args.epochs, loss1.item(), (end-start) * 2))
            print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!" % (epoch, args.epochs, loss2.item(), (end-start) * 2))
            # print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!" % (epoch, args.epochs, loss3.item(), (end-start) * 2))

            # model1.eval()
            model2.eval()
            # model3.eval()

            # classes = ('bareland', 'cropland', 'forest', 'impervious', 'shrub', 'water')
            classes = ('住宅区', '公共服务区域', '商业区', '工业区')  # ('住宅区', '公共服务区域', '商业区', '城市绿地', '工业区')
            # classes = ('Airport', 'BareLand', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential', 'Desert', 'Farmland', 'Forest', 'Industrial', 'Meadow', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Pond', 'Port', 'RailwayStation', 'Resort', 'River', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct')
            # classes = ('1 industrial land', '10 shrub land', '11 natural grassland', '12 artificial grassland', '13 river', '14 lake', '15 pond', '2 urban residential', '3 rural residential', '4 traffic land', '5 paddy field', '6 irrigated land', '7 dry cropland', '8 garden plot', '9 arbor woodland')
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

                    outputs2 = model2(images, poi_dense_h, poi_dense_w)
                    _2, predicted2 = torch.max(outputs2, 1)
                    c2 = (predicted2 == labels).squeeze()
                    for label_idx in range(len(labels)):
                        label = labels[label_idx]
                        class_correct2[label] += c2[label_idx].item()
                        class_total2[label] += 1
                    total_2 += labels.size(0)
                    # add correct
                    correct_prediction_2 += (predicted2 == labels).sum().item()


            for i in range(args.num_class):
                print('Model ResNeXt - Accuracy of %5s : %2d %%: Correct Num: %d in Total Num: %d' % (
                    classes[i], 100 * class_correct2[i] / class_total2[i], class_correct2[i], class_total2[i]))
            acc_2 = correct_prediction_2 / total_2
            print("Total Acc Model ResNeXt: %.4f" % (correct_prediction_2 / total_2))
            print('----------------------------------------------------')

        if acc_2 > best_acc_2:
            print('save new best acc_2', acc_2)
            torch.save(model2, os.path.join(args.model_path, 'RSIMG-UFZ-teacher-resnext50_32x4d_4class.pth'))
            best_acc_2 = acc_2
            best_epoch = epoch
        # if acc_3 > best_acc_3:
        #     print('save new best acc_3', acc_3)
        #     torch.save(model3, os.path.join(args.model_path, 'AID-30-teacher-densenet121-%s.pth' % (args.model_name)))
        #     best_acc_3 = acc_3
    # print("Model save to %s."%(os.path.join(args.model_path, 'UFZ-teacher-model-%s.pth' % (args.model_name))))
    # print('save new best acc_1', best_acc_1)
    print('save new best acc_2', best_acc_2, best_epoch)
    # print('save new best acc_3', best_acc_3)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train hyper-parameter')
    parser.add_argument("--num_class", default=4, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    # parser.add_argument("--net", default='ResNet50', type=str)
    # parser.add_argument("--depth", default=50, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    # parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--model_name", default='', type=str)
    parser.add_argument("--model_path", default='./model-UFZ', type=str)
    parser.add_argument("--pretrained", default=False, type=bool)
    parser.add_argument("--pretrained_model", default='./ImageNet-models/resnext50_32x4d-7cdf4587.pth', type=str)
    args = parser.parse_args()

    main(args)
