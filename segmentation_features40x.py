#!/usr/bin/env python
# coding: utf-8


# !/usr/bin/env python
# coding: utf-8


import openslide

import torch
from torch import nn
from torchvision import transforms, models

import os
import copy
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import os

WSI_TILE_CLASSIFICATION_PATH = './WSI_tile_classification'
filename = os.listdir(WSI_TILE_CLASSIFICATION_PATH)
# print(len(filename))
os.mkdir('./9mapping')
tissue_colors = {
    0: (0, 0, 0),
    1: (0, 0, 0),
    2: (0, 0, 0),
    3: (0, 50, 0),
    4: (50, 50, 50),
    5: (50, 50, 50),
    6: (50, 50, 50),
    7: (200, 200, 0),
    8: (50, 50, 50)
}

for i in range(0, len(filename)):
    # Read csv
    dataframe = pd.read_csv(os.path.join(WSI_TILE_CLASSIFICATION_PATH, filename[i]))
    pathology = dataframe.at[0, 'pathology']
    width = dataframe.at[0, 'width(tile)']
    height = dataframe.at[0, 'height(tile)']
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    rows, cols, channels = image_array.shape

    # Color tum block
    number_of_tile = len(dataframe.index)
    x_list = list(dataframe['x(tile)'])
    y_list = list(dataframe['y(tile)'])
    tissue_list = list(dataframe['tissue'])

    for j in range(0, number_of_tile):
        tissue_type = tissue_list[j]
        if tissue_type in tissue_colors:
            color = tissue_colors[tissue_type]
            image_array[y_list[j] - 1, x_list[j] - 1] = color

    image_mapping = Image.fromarray(image_array)
    image_mapping.save(os.path.join('./9mapping', pathology + '.png'))

os.mkdir('./tum_mapping')

for i in range(0, len(filename)):
    # Read csv
    dataframe = pd.read_csv(os.path.join(WSI_TILE_CLASSIFICATION_PATH, filename[i]))
    pathology = dataframe.at[0, 'pathology']
    width = dataframe.at[0, 'width(tile)']
    height = dataframe.at[0, 'height(tile)']

    # Create blank image
    image = Image.new('L', (width, height), 0)
    image_array = np.array(image)
    rows, cols = image_array.shape

    # Color tum block
    number_of_tile = len(dataframe.index)
    x_list = list(dataframe['x(tile)'])
    y_list = list(dataframe['y(tile)'])
    tissue_list = list(dataframe['tissue'])

    for j in range(0, number_of_tile):
        if tissue_list[j] == 8:
            image_array[y_list[j] - 1, x_list[j] - 1] = 255

    image_mapping = Image.fromarray(image_array)
    image_mapping.save(os.path.join('./tum_mapping', pathology + '.png'))

TUM_MAPPING_PATH = './tum_mapping'
TMfilename = os.listdir(TUM_MAPPING_PATH)
os.mkdir('./tum_closed')

for i in range(0, len(TMfilename)):
    # Read image
    binary = cv2.imread(os.path.join(TUM_MAPPING_PATH, TMfilename[i]), cv2.IMREAD_GRAYSCALE)

    # Closing : connect objects that were mistakenly divided into many small pieces
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Structure Element
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Save image
    cv2.imwrite(os.path.join('./tum_closed', TMfilename[i][0:22] + '.png'), closed)  # 23改为22

os.mkdir('./lym_mapping')

for i in range(0, len(filename)):
    # Read csv
    dataframe = pd.read_csv(os.path.join(WSI_TILE_CLASSIFICATION_PATH, filename[i]))
    pathology = dataframe.at[0, 'pathology']
    width = dataframe.at[0, 'width(tile)']
    height = dataframe.at[0, 'height(tile)']

    # Create blank image
    image = Image.new('L', (width, height), 0)
    image_array = np.array(image)
    rows, cols = image_array.shape

    # Color lym block
    number_of_tile = len(dataframe.index)
    x_list = list(dataframe['x(tile)'])
    y_list = list(dataframe['y(tile)'])
    tissue_list = list(dataframe['tissue'])

    for j in range(0, number_of_tile):
        if tissue_list[j] == 3:
            image_array[y_list[j] - 1, x_list[j] - 1] = 255

    image_mapping = Image.fromarray(image_array)
    image_mapping.save(os.path.join('./lym_mapping', pathology + '.png'))

lym_MAPPING_PATH = './lym_mapping'

os.mkdir('./stroma_mapping')

for i in range(0, len(filename)):
    # Read csv
    dataframe = pd.read_csv(os.path.join(WSI_TILE_CLASSIFICATION_PATH, filename[i]))
    pathology = dataframe.at[0, 'pathology']
    width = dataframe.at[0, 'width(tile)']
    height = dataframe.at[0, 'height(tile)']

    # Create blank image
    image = Image.new('L', (width, height), 0)
    image_array = np.array(image)
    rows, cols = image_array.shape

    # Color normal block
    number_of_tile = len(dataframe.index)
    x_list = list(dataframe['x(tile)'])
    y_list = list(dataframe['y(tile)'])
    tissue_list = list(dataframe['tissue'])

    for j in range(0, number_of_tile):
        if tissue_list[j] == 7:
            image_array[y_list[j] - 1, x_list[j] - 1] = 255

    image_mapping = Image.fromarray(image_array)
    image_mapping.save(os.path.join('./stroma_mapping', pathology + '.png'))

os.mkdir('./Normal_mapping')

for i in range(0, len(filename)):
    # Read csv
    dataframe = pd.read_csv(os.path.join(WSI_TILE_CLASSIFICATION_PATH, filename[i]))
    pathology = dataframe.at[0, 'pathology']
    width = dataframe.at[0, 'width(tile)']
    height = dataframe.at[0, 'height(tile)']

    # Create blank image
    image = Image.new('L', (width, height), 0)
    image_array = np.array(image)
    rows, cols = image_array.shape

    # Color stroma block
    number_of_tile = len(dataframe.index)
    x_list = list(dataframe['x(tile)'])
    y_list = list(dataframe['y(tile)'])
    tissue_list = list(dataframe['tissue'])

    for j in range(0, number_of_tile):
        if tissue_list[j] == 6:
            image_array[y_list[j] - 1, x_list[j] - 1] = 255

    image_mapping = Image.fromarray(image_array)
    image_mapping.save(os.path.join('./Normal_mapping', pathology + '.png'))

STROMA_MAPPING_PATH = './stroma_mapping'
SMfilename = os.listdir(STROMA_MAPPING_PATH)
os.mkdir('./stroma_closed')

for i in range(0, len(SMfilename)):
    # Read image
    binary = cv2.imread(os.path.join(STROMA_MAPPING_PATH, SMfilename[i]), cv2.IMREAD_GRAYSCALE)

    # Closing : connect objects that were mistakenly divided into many small pieces
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Structure Element
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Save image
    cv2.imwrite(os.path.join('./stroma_closed', SMfilename[i][0:22] + '.png'), closed)  # 23改为22

Normal_MAPPING_PATH = './Normal_mapping'
TMfilename = os.listdir(Normal_MAPPING_PATH)
os.mkdir('./Normal_closed')
for i in range(0, len(SMfilename)):
    # Read image
    binary = cv2.imread(os.path.join(Normal_MAPPING_PATH, SMfilename[i]), cv2.IMREAD_GRAYSCALE)

    # Closing : connect objects that were mistakenly divided into many small pieces
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Structure Element
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Save image
    cv2.imwrite(os.path.join('./Normal_closed', SMfilename[i][0:22] + '.png'), closed)  # 23改为22

os.mkdir('./DEB_mapping')

for i in range(0, len(filename)):
    # Read csv
    dataframe = pd.read_csv(os.path.join(WSI_TILE_CLASSIFICATION_PATH, filename[i]))
    pathology = dataframe.at[0, 'pathology']
    width = dataframe.at[0, 'width(tile)']
    height = dataframe.at[0, 'height(tile)']

    # Create blank image
    image = Image.new('L', (width, height), 0)
    image_array = np.array(image)
    rows, cols = image_array.shape

    # Color stroma block
    number_of_tile = len(dataframe.index)
    x_list = list(dataframe['x(tile)'])
    y_list = list(dataframe['y(tile)'])
    tissue_list = list(dataframe['tissue'])

    for j in range(0, number_of_tile):
        if tissue_list[j] == 0:
            image_array[y_list[j] - 1, x_list[j] - 1] = 255

    image_mapping = Image.fromarray(image_array)
    image_mapping.save(os.path.join('./DEB_mapping', pathology + '.png'))

DEB_MAPPING_PATH = './DEB_mapping'
TMfilename = os.listdir(DEB_MAPPING_PATH)
os.mkdir('./DEB_closed')
for i in range(0, len(SMfilename)):
    # Read image
    binary = cv2.imread(os.path.join(DEB_MAPPING_PATH, SMfilename[i]), cv2.IMREAD_GRAYSCALE)

    # Closing : connect objects that were mistakenly divided into many small pieces
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Structure Element
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Save image
    cv2.imwrite(os.path.join('./DEB_closed', SMfilename[i][0:22] + '.png'), closed)  # 23改为22

os.mkdir('./MUC_mapping')

for i in range(0, len(filename)):
    # Read csv
    dataframe = pd.read_csv(os.path.join(WSI_TILE_CLASSIFICATION_PATH, filename[i]))
    pathology = dataframe.at[0, 'pathology']
    width = dataframe.at[0, 'width(tile)']
    height = dataframe.at[0, 'height(tile)']

    # Create blank image
    image = Image.new('L', (width, height), 0)
    image_array = np.array(image)
    rows, cols = image_array.shape

    # Color stroma block
    number_of_tile = len(dataframe.index)
    x_list = list(dataframe['x(tile)'])
    y_list = list(dataframe['y(tile)'])
    tissue_list = list(dataframe['tissue'])

    for j in range(0, number_of_tile):
        if tissue_list[j] == 4:
            image_array[y_list[j] - 1, x_list[j] - 1] = 255

    image_mapping = Image.fromarray(image_array)
    image_mapping.save(os.path.join('./MUC_mapping', pathology + '.png'))

MUC_MAPPING_PATH = './MUC_mapping'
TMfilename = os.listdir(MUC_MAPPING_PATH)
os.mkdir('./MUC_closed')
for i in range(0, len(SMfilename)):
    # Read image
    binary = cv2.imread(os.path.join(MUC_MAPPING_PATH, SMfilename[i]), cv2.IMREAD_GRAYSCALE)

    # Closing : connect objects that were mistakenly divided into many small pieces
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Structure Element
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Save image
    cv2.imwrite(os.path.join('./MUC_closed', SMfilename[i][0:22] + '.png'), closed)  # 23改为22

os.mkdir('./MUS_mapping')

for i in range(0, len(filename)):
    # Read csv
    dataframe = pd.read_csv(os.path.join(WSI_TILE_CLASSIFICATION_PATH, filename[i]))
    pathology = dataframe.at[0, 'pathology']
    width = dataframe.at[0, 'width(tile)']
    height = dataframe.at[0, 'height(tile)']

    # Create blank image
    image = Image.new('L', (width, height), 0)
    image_array = np.array(image)
    rows, cols = image_array.shape

    # Color stroma block
    number_of_tile = len(dataframe.index)
    x_list = list(dataframe['x(tile)'])
    y_list = list(dataframe['y(tile)'])
    tissue_list = list(dataframe['tissue'])

    for j in range(0, number_of_tile):
        if tissue_list[j] == 5:
            image_array[y_list[j] - 1, x_list[j] - 1] = 255

    image_mapping = Image.fromarray(image_array)
    image_mapping.save(os.path.join('./MUS_mapping', pathology + '.png'))

MUS_MAPPING_PATH = './MUS_mapping'
TMfilename = os.listdir(MUS_MAPPING_PATH)
os.mkdir('./MUS_closed')
for i in range(0, len(SMfilename)):
    # Read image
    binary = cv2.imread(os.path.join(MUS_MAPPING_PATH, SMfilename[i]), cv2.IMREAD_GRAYSCALE)

    # Closing : connect objects that were mistakenly divided into many small pieces
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Structure Element
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Save image
    cv2.imwrite(os.path.join('./MUS_closed', SMfilename[i][0:22] + '.png'), closed)  # 23改为22

pathology = []

for i in range(0, len(TMfilename)):
    pathology.append(TMfilename[i][0:22])  # 23改为22

max_tumor_area = []

tum_number = []
tum_labels = []
tum_stats = []

for i in range(0, len(pathology)):
    # Read image
    closed = cv2.imread(os.path.join('./tum_closed', pathology[i] + '.png'), cv2.IMREAD_GRAYSCALE)

    # Connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8, ltype=cv2.CV_32S)
    tum_number.append(num_labels - 1)
    tum_labels.append(labels)
    tum_stats.append(stats)  # x,y,width,height,area

    # max_tumor_area
    if num_labels > 1:
        max_tumor_area_label = np.argmax(stats[1:, 4]) + 1
        max_tumor_area.append(stats[max_tumor_area_label, 4])
    else:
        max_tumor_area.append(0)

max_Normal_area = []
Normal_number = []
Normal_labels = []
Normal_stats = []
for i in range(0, len(pathology)):
    closed = cv2.imread(os.path.join('./Normal_closed', pathology[i] + '.png'), cv2.IMREAD_GRAYSCALE)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8, ltype=cv2.CV_32S)
    Normal_number.append(num_labels - 1)
    Normal_labels.append(labels)
    Normal_stats.append(stats)  # x,y,width,height,area

    if num_labels > 1:
        max_Normal_area_label = np.argmax(stats[1:, 4]) + 1
        max_Normal_area.append(stats[max_Normal_area_label, 4])
    else:
        max_Normal_area.append(0)

max_DEB_area = []
DEB_number = []
DEB_labels = []
DEB_stats = []
for i in range(0, len(pathology)):
    closed = cv2.imread(os.path.join('./DEB_closed', pathology[i] + '.png'), cv2.IMREAD_GRAYSCALE)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8, ltype=cv2.CV_32S)
    DEB_number.append(num_labels - 1)
    DEB_labels.append(labels)
    DEB_stats.append(stats)  # x,y,width,height,area

    if num_labels > 1:
        max_DEB_area_label = np.argmax(stats[1:, 4]) + 1
        max_DEB_area.append(stats[max_DEB_area_label, 4])
    else:
        max_DEB_area.append(0)

max_MUC_area = []
MUC_number = []
MUC_labels = []
MUC_stats = []
for i in range(0, len(pathology)):
    closed = cv2.imread(os.path.join('./MUC_closed', pathology[i] + '.png'), cv2.IMREAD_GRAYSCALE)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8, ltype=cv2.CV_32S)
    MUC_number.append(num_labels - 1)
    MUC_labels.append(labels)
    MUC_stats.append(stats)  # x,y,width,height,area

    if num_labels > 1:
        max_MUC_area_label = np.argmax(stats[1:, 4]) + 1
        max_MUC_area.append(stats[max_MUC_area_label, 4])
    else:
        max_MUC_area.append(0)

max_MUS_area = []
MUS_number = []
MUS_labels = []
MUS_stats = []
for i in range(0, len(pathology)):
    closed = cv2.imread(os.path.join('./MUS_closed', pathology[i] + '.png'), cv2.IMREAD_GRAYSCALE)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8, ltype=cv2.CV_32S)
    MUS_number.append(num_labels - 1)
    MUS_labels.append(labels)
    MUS_stats.append(stats)  # x,y,width,height,area

    if num_labels > 1:
        max_MUS_area_label = np.argmax(stats[1:, 4]) + 1
        max_MUS_area.append(stats[max_MUS_area_label, 4])
    else:
        max_MUS_area.append(0)

lymphocyte_inside_tumor = []

for i in range(0, len(pathology)):
    # Read image
    tum_array = cv2.imread(os.path.join('./tum_closed', pathology[i] + '.png'), cv2.IMREAD_GRAYSCALE)
    lym_array = cv2.imread(os.path.join('./lym_mapping', pathology[i] + '.png'), cv2.IMREAD_GRAYSCALE)

    # lymphocyte_inside_tumor
    lit = 0
    for j in range(0, tum_array.shape[0]):
        for k in range(0, tum_array.shape[1]):
            if tum_array[j, k] == 255 and lym_array[j, k] == 255:
                lit = lit + 1

    lymphocyte_inside_tumor.append(lit)

lymphocyte_around_tumor = []

for i in range(0, len(pathology)):
    stats = copy.deepcopy(tum_stats[i])  # x,y,width,height,area
    tum_label = copy.deepcopy(tum_labels[i])  # 0,1,2...
    lym_array = cv2.imread(os.path.join('./lym_mapping', pathology[i] + '.png'), cv2.IMREAD_GRAYSCALE)  # 0,255
    check_array = np.zeros(lym_array.shape, np.uint8)

    # establish check_array
    for j in range(1, tum_number[i] + 1):
        # change the value of tumor j to 255, others to 0
        tumor = copy.deepcopy(tum_label)
        tumor[tumor != j] = 0
        tumor[tumor == j] = 255

        check_size = [20, 20]

        for k in range(0, tumor.shape[0]):
            for l in range(0, tumor.shape[1]):
                if tumor[k, l] == 255:
                    y_lower = k - check_size[0]
                    if y_lower < 0:
                        y_lower = 0
                    y_upper = k + check_size[0]
                    if y_upper > tumor.shape[0] - 1:
                        y_upper = tumor.shape[0] - 1

                    x_lower = l - check_size[1]
                    if x_lower < 0:
                        x_lower = 0
                    x_upper = l + check_size[1]
                    if x_upper > tumor.shape[1] - 1:
                        x_upper = tumor.shape[1] - 1

                    for m in range(y_lower, y_upper + 1):
                        for n in range(x_lower, x_upper + 1):
                            check_array[m, n] = 255

    check_array[tum_label != 0] = 0

    # lymphocyte_around_tumor
    lat = 0
    for j in range(0, check_array.shape[0]):
        for k in range(0, check_array.shape[1]):
            if check_array[j, k] == 255 and lym_array[j, k] == 255:
                lat = lat + 1

    lymphocyte_around_tumor.append(lat)

around_inside_ratio = []

for i in range(0, len(pathology)):
    around_inside_ratio.append((lymphocyte_around_tumor[i] + 1) / (lymphocyte_inside_tumor[i] + 1))

total_stroma_area = []

for i in range(0, len(pathology)):
    # Read image
    closed = cv2.imread(os.path.join('./stroma_closed', pathology[i] + '.png'), cv2.IMREAD_GRAYSCALE)

    # total_stroma_area
    total_stroma_area.append(np.count_nonzero(closed))

survival = pd.read_csv('./survival_COAD_survival.csv', index_col='sample')

tumor_stroma_ratio = []
for i in range(len(pathology)):
    tumor_area = max_tumor_area[i]
    stroma_area = total_stroma_area[i]  # 假设total_stroma_area已通过np.count_nonzero计算
    if stroma_area == 0:
        ratio = 0.0
    else:
        ratio = tumor_area / stroma_area
    tumor_stroma_ratio.append(ratio)

muc_stroma_ratio = []
for i in range(len(pathology)):
    muc_area = max_MUC_area[i]  # 假设已计算max_MUC_area
    stroma_area = total_stroma_area[i]
    if stroma_area == 0:
        ratio = 0.0
    else:
        ratio = muc_area / stroma_area
    muc_stroma_ratio.append(ratio)

mus_stroma_ratio = []
for i in range(len(pathology)):
    mus_area = max_MUS_area[i]  # 假设已计算max_MUS_area
    stroma_area = total_stroma_area[i]
    if stroma_area == 0:
        ratio = 0.0
    else:
        ratio = mus_area / stroma_area
    mus_stroma_ratio.append(ratio)

# 假设脂肪对应类别未在代码中处理，此处以DEB为例（原代码中DEB对应类别0）
deb_stroma_ratio = []
for i in range(len(pathology)):
    deb_area = max_DEB_area[i]  # 已计算max_DEB_area
    stroma_area = total_stroma_area[i]
    if stroma_area == 0:
        ratio = 0.0
    else:
        ratio = deb_area / stroma_area
    deb_stroma_ratio.append(ratio)

OS = []
OS_time = []

for i in range(0, len(pathology)):
    OS.append(survival.at[pathology[i][0:15], 'OS'])
    OS_time.append(survival.at[pathology[i][0:15], 'OS.time'])

df = pd.DataFrame({
    'pathology': pathology,
    'max_tumor_area': max_tumor_area,
    'lymphocyte_inside_tumor': lymphocyte_inside_tumor,
    'lymphocyte_around_tumor': lymphocyte_around_tumor,
    'around_inside_ratio': around_inside_ratio,
    'max_Normal_area': max_Normal_area,
    'max_DEB_area': max_DEB_area,
    'max_MUC_area': max_MUC_area,
    'max_MUS_area': max_MUS_area,
    'total_stroma_area': total_stroma_area,
    'tumor_stroma_ratio': tumor_stroma_ratio,
    'muc_stroma_ratio': muc_stroma_ratio,
    'mus_stroma_ratio': mus_stroma_ratio,
    'deb_stroma_ratio': deb_stroma_ratio,
    'OS': OS,
    'OS.time': OS_time
})

segmentation_feature = ['max_tumor_area',
                        'lymphocyte_inside_tumor',
                        'lymphocyte_around_tumor',
                        'around_inside_ratio',
                        'max_Normal_area',
                        'max_DEB_area',
                        'max_MUC_area',
                        'max_MUS_area',
                        'total_stroma_area',
                        'tumor_stroma_ratio',
                        'muc_stroma_ratio',
                        'mus_stroma_ratio',
                        'deb_stroma_ratio']
for feature in segmentation_feature:
    median_value = df[feature].median()
    df[feature] = df[feature].apply(lambda x: 1 if x > median_value else 0)
df.to_csv('./segmentation_features10xx.csv', index=False)