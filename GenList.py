# coding=utf-8
import os
import cv2
import random
import numpy
import sys

# if __name__ == "__main__":

# dict = {'Built-up': 0, 'bareland': 1, 'vegetation_coverage': 2, 'water': 3}
def label_of_directory(directory):
    """
    sorted for label indices
    return a dict for {'classes', 'range(len(classes))'}
    """
    classes = []
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            classes.append(subdir)

    num_classes = len(classes)
    class_indices = dict(zip(classes, range(len(classes))))
    #print(num_classes)#,class_indices(classes))
    return class_indices

rate = 0.9       #随机抽取10%的样本作为验证集
# root = '.\\GenDatasetAID\\'
root = '.\\GenShenzhenUFZ-8-2-4class\\train_data\\'
dict = label_of_directory(root)
print(dict)

Trainlist = []
Testlist = []
alllist = []
index = 0
# max_num = 80000

for folder in dict:
    img_list = [f for f in os.listdir(os.path.join(root, folder)) if not f.startswith('.')]
    for img in img_list:
        str0 = '%d\t%s\t%d\n' % (index, os.path.join(folder, img), dict[folder])
        index += 1
        alllist.append(str0)

Trainlist = alllist

Trainfile = open("./data/trainGenShenzhenUFZ-8-2-4class.txt", "w")
for str1 in Trainlist:
    Trainfile.write(str1)
Trainfile.close()

root = '.\\GenShenzhenUFZ-8-2-4class\\test_data\\'
# dict = label_of_directory(root)
# print(dict)

Trainlist = []
Testlist = []
alllist = []
index = 0
# max_num = 80000

for folder in dict:
    img_list = [f for f in os.listdir(os.path.join(root, folder)) if not f.startswith('.')]
    for img in img_list:
        str0 = '%d\t%s\t%d\n' % (index, os.path.join(folder, img), dict[folder])
        index += 1
        alllist.append(str0)

Testlist = alllist
Testfile = open("./data/testGenShenzhenUFZ-8-2-4class.txt", "w")
for str1 in Testlist:
    Testfile.write(str1)
Testfile.close()
