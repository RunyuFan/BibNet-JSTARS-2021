# -*- coding:utf-8 -*-
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torch

from osgeo import gdal
import os
import geopandas
import json

class POI23DateSet(data.Dataset):

    def __init__(self, root, lists, transforms=None, train=True, test=False):

        self.test = test
        # imgs = [os.path.join(root, img) for img in os.listdir(root)]
        self.category = ['住宅区', '工业区', '公共服务区域', '商业区']

        with open (lists, 'r') as f:
            lines = f.readlines()

        imgs = []
        labels = []
        pois = []

        dic = self.get_poi_dict()

        for line in lines:
            imgpath = os.path.join(root, line.split('\t')[1])
            imgpath_tif = os.path.join(imgpath.split('\\')[-2], imgpath.split('\\')[-1])
            imgs.append(imgpath)
            # print(imgpath, imgpath_tif, dic[imgpath_tif])
            labels.append(int(line.split('\t')[2])) # irrigated land_1.tif
            pois.append(dic[imgpath_tif])
        # pois = np.array(pois)
        # print(pois)
        self.imgs = imgs
        self.labels = labels
        self.pois = pois
        # self.area = np.array(self.pois)[:, -1]
        self.maxh = np.array(np.array(self.pois).max(axis=0))
        self.maxw = np.array(self.pois)[:, 0:23].max(axis=1)

        if transforms is None:

            self.transforms = T.Compose([
                # torchvision.transforms.Resize(256),
                # T.ToTensor()
                # T.Resize(224),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
                # T.ToPILImage(),
                T.Resize((224, 224)),  # 缩放图片(Image)到(h,w)
                # T.RandomHorizontalFlip(p=0.3),
                # T.RandomVerticalFlip(p=0.3),
                # T.RandomCrop(size=224),
                # T.RandomRotation(180),
                # T.RandomHorizontalFlip(), #水平翻转，注意不是所有图片都适合，比如车牌
                # T.CenterCrop(224),  # 从图片中间切出224*224的图片
                # T.RandomCrop(224),  #随机裁剪
                T.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]
                # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化至[-1, 1]，规定均值和标准差
            ])
            self.transforms_poi = T.Compose([
                # torchvision.transforms.Resize(256),
                # T.ToTensor()
                # T.Resize(224),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
                # T.ToPILImage(),
                # T.Resize((224, 224)),  # 缩放图片(Image)到(h,w)
                # T.RandomHorizontalFlip(p=0.3),
                # T.RandomVerticalFlip(p=0.3),
                # T.RandomCrop(size=224),
                # T.RandomRotation(180),
                # T.RandomHorizontalFlip(), #水平翻转，注意不是所有图片都适合，比如车牌
                # T.CenterCrop(224),  # 从图片中间切出224*224的图片
                # T.RandomCrop(224),  #随机裁剪
                T.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]
                # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化至[-1, 1]，规定均值和标准差
            ])
        else:
            self.transforms = transforms



    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        label = self.labels[index]
        poi = self.pois[index]
        data = Image.open(img_path).convert('RGB')
        data = self.transforms(data)
        # print(self.maxh)

        maxh_div = [1 if i == 0 else i for i in self.maxh]
        poi_dense_h = poi / maxh_div

        div_num = self.maxw[index]
        if div_num == 0:
            div_num = 1
        poi_dense_w = [x / div_num for x in poi]
        # poi_dense_area = [x / self.area[index] for x in poi]
        poi_dense_h = np.array(poi_dense_h)[0:23].astype(float).reshape((1, -1))
        poi_dense_w = np.array(poi_dense_w)[0:23].astype(float).reshape((1, -1))
        # poi_dense_area = np.array(poi_dense_area)[0:5].astype(float).reshape((1, -1))
        poi_dense_h = self.transforms_poi(poi_dense_h)
        poi_dense_w = self.transforms_poi(poi_dense_w)
        poi_dense_h = poi_dense_h.view(1, 23)
        poi_dense_w = poi_dense_w.view(1, 23)
        # poi_dense_area = self.transforms_poi(poi_dense_area)
        return data, label, poi_dense_h, poi_dense_w

    def __len__(self):
        return len(self.imgs)

    def generator_list_of_path(self, path):
        image_list = []
        for image in os.listdir(path):
            # print(path)
            # print(image)
            if not image == '.DS_Store' and 'shp' == image.split('.')[-1]:
                image_list.append(os.path.join(path, image))
        return image_list

    def get_poi_dict(self):
        img_list = []
        poi_list = []
        for idx in range(4):
            shp_path = '.\\UFZ_shp_5class_23poi\\' + self.category[idx]
            # print(shp_path)
            input_shape = self.generator_list_of_path(shp_path)
            null_list = []
            for shp_file in input_shape:
                # print(shp_file)
                gdf = geopandas.GeoDataFrame.from_file(shp_file, encoding='utf-8')  #fangjia
                # imgpath = 'C:\\Users\\Fly_D\\Desktop\\DS-UFZ\\GenShenzhenUFZ_4class\\' + category[idx] + '\\' + str(gdf.idx) + '.tif'
                poi = gdf.values[:, -25:-1]
                # print(poi)
                imgpath = os.path.join(self.category[idx], str(poi[:, -1][0]) + '.tif')
                img_list.append(imgpath)
                poi_list.append(poi[0])
        dic = dict(zip(img_list, poi_list))
        return dic

class CarDateSet(data.Dataset):

    def __init__(self, root, lists, transforms=None, train=True, test=False):

        self.test = test
        # imgs = [os.path.join(root, img) for img in os.listdir(root)]
        self.category = ['住宅区', '工业区', '公共服务区域', '商业区']

        with open (lists, 'r') as f:
            lines = f.readlines()

        imgs = []
        labels = []
        pois = []

        dic = self.get_poi_dict()

        for line in lines:
            imgpath = os.path.join(root, line.split('\t')[1])
            imgpath_tif = os.path.join(imgpath.split('\\')[-2], imgpath.split('\\')[-1])
            imgs.append(imgpath)
            # print(imgpath, imgpath_tif, dic[imgpath_tif])
            labels.append(int(line.split('\t')[2])) # irrigated land_1.tif
            pois.append(dic[imgpath_tif])
        # pois = np.array(pois)
        # print(pois)
        self.imgs = imgs
        self.labels = labels
        self.pois = pois
        self.area = np.array(self.pois)[:, -1]
        self.maxh = np.array(np.array(self.pois).max(axis=0))
        self.maxw = np.array(self.pois)[:, 0:4].max(axis=1)

        if transforms is None:

            self.transforms = T.Compose([
                # torchvision.transforms.Resize(256),
                # T.ToTensor()
                # T.Resize(224),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
                # T.ToPILImage(),
                T.Resize((224, 224)),  # 缩放图片(Image)到(h,w)
                # T.RandomHorizontalFlip(p=0.3),
                # T.RandomVerticalFlip(p=0.3),
                # T.RandomCrop(size=224),
                # T.RandomRotation(180),
                # T.RandomHorizontalFlip(), #水平翻转，注意不是所有图片都适合，比如车牌
                # T.CenterCrop(224),  # 从图片中间切出224*224的图片
                # T.RandomCrop(224),  #随机裁剪
                T.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]
                # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化至[-1, 1]，规定均值和标准差
            ])
            self.transforms_poi = T.Compose([
                # torchvision.transforms.Resize(256),
                # T.ToTensor()
                # T.Resize(224),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
                # T.ToPILImage(),
                # T.Resize((224, 224)),  # 缩放图片(Image)到(h,w)
                # T.RandomHorizontalFlip(p=0.3),
                # T.RandomVerticalFlip(p=0.3),
                # T.RandomCrop(size=224),
                # T.RandomRotation(180),
                # T.RandomHorizontalFlip(), #水平翻转，注意不是所有图片都适合，比如车牌
                # T.CenterCrop(224),  # 从图片中间切出224*224的图片
                # T.RandomCrop(224),  #随机裁剪
                T.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]
                # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化至[-1, 1]，规定均值和标准差
            ])
        else:
            self.transforms = transforms



    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        label = self.labels[index]
        poi = self.pois[index]
        data = Image.open(img_path).convert('RGB')
        data = self.transforms(data)
        poi_dense_h = poi / self.maxh
        div_num = self.maxw[index]
        if div_num == 0:
            div_num = 1
        poi_dense_w = [x / div_num for x in poi]
        # poi_dense_area = [x / self.area[index] for x in poi]
        poi_dense_h = np.array(poi_dense_h)[0:4].astype(float).reshape((1, -1))
        poi_dense_w = np.array(poi_dense_w)[0:4].astype(float).reshape((1, -1))
        # poi_dense_area = np.array(poi_dense_area)[0:5].astype(float).reshape((1, -1))
        poi_dense_h = self.transforms_poi(poi_dense_h)
        poi_dense_w = self.transforms_poi(poi_dense_w)
        poi_dense_h = poi_dense_h.view(1, 4)
        poi_dense_w = poi_dense_w.view(1, 4)
        # poi_dense_area = self.transforms_poi(poi_dense_area)
        return data, label, poi_dense_h, poi_dense_w

    def __len__(self):
        return len(self.imgs)

    def generator_list_of_path(self, path):
        image_list = []
        for image in os.listdir(path):
            # print(path)
            # print(image)
            if not image == '.DS_Store' and 'shp' == image.split('.')[-1]:
                image_list.append(os.path.join(path, image))
        return image_list

    def get_poi_dict(self):
        img_list = []
        poi_list = []
        for idx in range(4):
            shp_path = '.\\UFZ_shp_5class\\' + self.category[idx]
            # print(shp_path)
            input_shape = self.generator_list_of_path(shp_path)
            null_list = []
            for shp_file in input_shape:
                # print(shp_file)
                gdf = geopandas.GeoDataFrame.from_file(shp_file, encoding='utf-8')  #fangjia
                # imgpath = 'C:\\Users\\Fly_D\\Desktop\\DS-UFZ\\GenShenzhenUFZ_4class\\' + category[idx] + '\\' + str(gdf.idx) + '.tif'
                poi = gdf.values[:, -7:-1]
                # print(poi)
                imgpath = os.path.join(self.category[idx], str(poi[:, 4][0]) + '.tif')
                img_list.append(imgpath)
                poi_list.append(poi[0])
        dic = dict(zip(img_list, poi_list))
        return dic

class TriDateSet(data.Dataset):

    def __init__(self, root, lists, transforms=None, train=True, test=False):

        self.test = test
        # imgs = [os.path.join(root, img) for img in os.listdir(root)]
        self.category = ['住宅区', '工业区', '公共服务区域', '商业区']

        with open (lists, 'r') as f:
            lines = f.readlines()

        imgs = []
        labels = []
        pois = []
        pops = []
        hps = []
        nls = []

        dic = self.get_poi_dict()

        for line in lines:
            imgpath = os.path.join(root, line.split('\t')[1])
            imgpath_tif = os.path.join(imgpath.split('\\')[-2], imgpath.split('\\')[-1])
            pop_path = os.path.join(r'C:\Users\Fly_D\Desktop\DS-UFZ\GenShenzhenUFZ_5class_pop_2020', imgpath_tif)
            imgs.append(imgpath)
            pops.append(pop_path)
            # print(imgpath, imgpath_tif, pop_path, dic[imgpath_tif])
            labels.append(int(line.split('\t')[2])) # irrigated land_1.tif
            pois.append(dic[imgpath_tif])
            hp_path = os.path.join(r'C:\Users\Fly_D\Desktop\DS-UFZ\GenShenzhenUFZ_5class_hp_2020', imgpath_tif)
            hps.append(hp_path)
            nls_path = os.path.join(r'C:\Users\Fly_D\Desktop\DS-UFZ\GenShenzhenUFZ_5class_nl_2020', imgpath_tif)
            nls.append(nls_path)
        # pois = np.array(pois)
        # print(pois)
        self.pops = pops
        self.nls = nls
        self.hps = hps
        self.imgs = imgs
        self.labels = labels
        self.pois = pois
        self.area = np.array(self.pois)[:, -1]
        self.maxh = np.array(np.array(self.pois).max(axis=0))
        self.maxw = np.array(self.pois)[:, 0:4].max(axis=1)

        if transforms is None:

            self.transforms = T.Compose([
                # torchvision.transforms.Resize(256),
                # T.ToTensor()
                # T.Resize(224),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
                # T.ToPILImage(),
                T.Resize((224, 224)),  # 缩放图片(Image)到(h,w)
                # T.RandomHorizontalFlip(p=0.3),
                # T.RandomVerticalFlip(p=0.3),
                # T.RandomCrop(size=224),
                # T.RandomRotation(180),
                # T.RandomHorizontalFlip(), #水平翻转，注意不是所有图片都适合，比如车牌
                # T.CenterCrop(224),  # 从图片中间切出224*224的图片
                # T.RandomCrop(224),  #随机裁剪
                T.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]
                # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化至[-1, 1]，规定均值和标准差
            ])
            self.transforms_pop = T.Compose([
                # torchvision.transforms.Resize(256),
                # T.ToTensor()
                # T.Resize(224),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
                # T.ToPILImage(),
                T.Resize((224, 224)),  # 缩放图片(Image)到(h,w)
                # T.RandomHorizontalFlip(p=0.3),
                # T.RandomVerticalFlip(p=0.3),
                # T.RandomCrop(size=224),
                # T.RandomRotation(180),
                # T.RandomHorizontalFlip(), #水平翻转，注意不是所有图片都适合，比如车牌
                # T.CenterCrop(224),  # 从图片中间切出224*224的图片
                # T.RandomCrop(224),  #随机裁剪
                T.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]
                # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化至[-1, 1]，规定均值和标准差
            ])
            self.transforms_poi = T.Compose([
                # torchvision.transforms.Resize(256),
                # T.ToTensor()
                # T.Resize(224),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
                # T.ToPILImage(),
                # T.Resize((224, 224)),  # 缩放图片(Image)到(h,w)
                # T.RandomHorizontalFlip(p=0.3),
                # T.RandomVerticalFlip(p=0.3),
                # T.RandomCrop(size=224),
                # T.RandomRotation(180),
                # T.RandomHorizontalFlip(), #水平翻转，注意不是所有图片都适合，比如车牌
                # T.CenterCrop(224),  # 从图片中间切出224*224的图片
                # T.RandomCrop(224),  #随机裁剪
                T.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]
                # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化至[-1, 1]，规定均值和标准差
            ])
        else:
            self.transforms = transforms



    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        # pop_path = self.pops[index]
        label = self.labels[index]
        poi = self.pois[index]
        data = Image.open(img_path).convert('RGB')
        data = self.transforms(data)
        pop_path = self.pops[index]
        pop_data = Image.open(pop_path)
        # print(pop_data)
        pop_data = self.transforms_pop(pop_data)

        nl_path = self.nls[index]
        nl_data = Image.open(nl_path)
        # print(pop_data.shape)
        nl_data = self.transforms_pop(nl_data)

        hp_path = self.hps[index]
        hp_data = Image.open(hp_path)
        # print(pop_data)
        hp_data = self.transforms_pop(hp_data)

        data_stack = torch.cat((hp_data, nl_data),0)
        # data_stack = data_stack.squeeze()
        # print(data_stack.shape)
        data_stack = torch.cat((data_stack, pop_data),0)
        # data_stack = data_stack.squeeze()
        # print(data_stack.shape)
        # data_stack = self.transforms_pop(data_stack)

        poi_dense_h = poi / self.maxh
        div_num = self.maxw[index]
        if div_num == 0:
            div_num = 1
        poi_dense_w = [x / div_num for x in poi]
        # poi_dense_area = [x / self.area[index] for x in poi]
        poi_dense_h = np.array(poi_dense_h)[0:4].astype(float).reshape((1, -1))
        poi_dense_w = np.array(poi_dense_w)[0:4].astype(float).reshape((1, -1))
        # poi_dense_area = np.array(poi_dense_area)[0:5].astype(float).reshape((1, -1))
        # print(poi_dense_h.shape)
        poi_dense_h = self.transforms_poi(poi_dense_h)
        poi_dense_w = self.transforms_poi(poi_dense_w)
        # print(poi_dense_h.shape)
        poi_dense_h = poi_dense_h.view(1, 4)
        poi_dense_w = poi_dense_w.view(1, 4)
        # print(poi_dense_h.shape)
        # poi_dense_area = self.transforms_poi(poi_dense_area)
        return data, hp_data, label, poi_dense_h, poi_dense_w

    def __len__(self):
        return len(self.imgs)

    def generator_list_of_path(self, path):
        image_list = []
        for image in os.listdir(path):
            # print(path)
            # print(image)
            if not image == '.DS_Store' and 'shp' == image.split('.')[-1]:
                image_list.append(os.path.join(path, image))
        return image_list

    def get_poi_dict(self):
        img_list = []
        poi_list = []
        for idx in range(4):
            shp_path = '.\\UFZ_shp_5class\\' + self.category[idx]
            # print(shp_path)
            input_shape = self.generator_list_of_path(shp_path)
            null_list = []
            for shp_file in input_shape:
                # print(shp_file)
                gdf = geopandas.GeoDataFrame.from_file(shp_file, encoding='utf-8')  #fangjia
                # imgpath = 'C:\\Users\\Fly_D\\Desktop\\DS-UFZ\\GenShenzhenUFZ_4class\\' + category[idx] + '\\' + str(gdf.idx) + '.tif'
                poi = gdf.values[:, -7:-1]
                # print(poi)
                imgpath = os.path.join(self.category[idx], str(poi[:, 4][0]) + '.tif')
                img_list.append(imgpath)
                poi_list.append(poi[0])
        dic = dict(zip(img_list, poi_list))
        return dic

if __name__ == '__main__':

    dataset =  TriDateSet('.\\GenShenzhenUFZ-8-2-4class\\train_data\\', './data/trainGenShenzhenUFZ-8-2-4class.txt', transforms=None)
    data, pop_data, label, poi_dense_h, poi_dense_w = dataset[0]  # 相当于调用dataset.__getitem__(0)
    print(data.shape, pop_data.shape, label, poi_dense_h.shape, poi_dense_w)
