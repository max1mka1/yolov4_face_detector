#!/usr/bin/env python
# coding: utf-8

# ## This is a YOLOv4 training pipeline with Pytorch.
# I use coco pre-trained weights.
# Have fun and feel free to leave any comment!

# ## Reference
# https://github.com/Tianxiaomo/pytorch-YOLOv4
# https://www.kaggle.com/orkatz2/yolov5-train

# In[20]:


# !pip install torch==1.4.0 torchvision==0.5.0
# !git clone https://github.com/Tianxiaomo/pytorch-YOLOv4
# !rm ./* -r
# !cp -r pytorch-YOLOv4/* ./
# !pip install -U -r requirements.txt


# In[21]:


import os
import random
import numpy as np
import pandas as pd
#from tqdm.notebook import tqdm  # tqdm_notebook as tqdm
from tqdm import tqdm
#from tqdm.contrib.concurrent import process_map
import shutil as sh
import cv2
from PIL import Image


def convert_dataset_to_coco(dataset_name: str = 'widerface', det_threshold: int = 3):

    PATH = os.getcwd()
    df = ''
    if dataset_name == 'wheat':
        # Wheat train.csv sample
        df = pd.read_csv(os.path.join(PATH, 'data', dataset_name, 'train.csv'))  # dataset_name
        bboxs = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
        for i, column in enumerate(['x', 'y', 'w', 'h']):
            df[column] = bboxs[:,i]
        df.drop(columns=['bbox'], inplace=True)
        df['x1'] = df['x'] + df['w']
        df['y1'] = df['y'] + df['h']
        df['classes'] = 0
        df = df[['image_id','x', 'y', 'w', 'h','x1','y1','classes']]
        print(df.shape)
        print(df.head())
        print(f"Unique classes: {df['classes'].unique()}")
    else:
        modes = ['train', 'test', 'val']
        label_path = dict([(mode, os.path.join(PATH, 'data', dataset_name, mode, 'label.txt')) for mode in modes])

        data_dict = {}
        for mode in modes:
            data_dict[mode] = {'label_path': label_path[mode]}

        data = {'image_id': [], 'x': [], 'y': [], 'w': [], 'h': [],
                     'x1': [], 'y1': [], 'classes': [], 'landmarks': []}

        def txt_to_list(path_to_file: str, mode: str):
            file = open(path_to_file,'r')
            lines = file.readlines()
            isFirst = True
            labels = []
            words = []
            imgs_path = []
            anno_folder = os.path.join(PATH, 'data', dataset_name)
            for line in lines:
                line = line.rstrip()
                if line.startswith('#'):
                    if isFirst is True:
                        isFirst = False
                    else:
                        labels_copy = labels.copy()
                        words.append(labels_copy)
                        labels.clear()
                    img_path = line[2:]
                    full_path = os.path.join(anno_folder, mode, 'images', img_path)
                    imgs_path.append(full_path)
                else:
                    line = line.split(' ')
                    label = [float(x) for x in line]
                    visible = True if (label[2] >= det_threshold and label[3] >= det_threshold) else False
                    if visible:
                        labels.append(label)
            words.append(labels)
            if mode == 'test':
                return imgs_path
            else:
                return words, labels, imgs_path


        for mode in modes:
            if mode != 'test':
                words, labels, imgs_path = txt_to_list(label_path[mode], mode)
                data_dict[mode]['words'] = words
                data_dict[mode]['labels'] = labels
                data_dict[mode]['imgs_path'] = imgs_path
            else:
                imgs_path = txt_to_list(label_path[mode], mode)
                # print(imgs_path)
                data_dict[mode]['imgs_path'] = imgs_path


        for mode in modes:
            if mode != 'test':
                new_words_list = []
                for i, word in enumerate(tqdm(data_dict[mode]['words'])):
                    for string in word:
                        new_words_list.append(tuple([data_dict[mode]['imgs_path'][i], string]))
                data_dict[mode]['new_words_list'] = new_words_list


        def convert_data(tuple_datum):
            path, string = tuple_datum
            file_name = f"{path.split('/')[-2]}/{path.split('/')[-1]}"  # .split('.')[0]
            x, y, w, h = string[0:4]
            x1, y1 = x + w, y + h
            landmarks = string[5:]
            data = {'image_id': file_name, 'x': x, 'y': y,
                    'w': w, 'h': h, 'x1': x1, 'y1': y1,
                    'classes': 0,
                    'landmarks': landmarks}
            return data


        def convert_data_val(path):
            file_name = f"{path.split('/')[-2]}/{path.split('/')[-1]}"  # .split('.')[0]
            data = {'image_id': file_name}
            return data


        for mode in modes:
            csv_path = os.path.join(PATH, 'data', dataset_name, mode, f'{mode}.csv')
            if mode != 'test':
                data_dict[mode]['csv_path'] = csv_path
                datum = []
                for path in tqdm(data_dict[mode]['new_words_list']):
                    datum.append(convert_data(path))
                df = pd.DataFrame(data=datum)
                df.to_csv(csv_path)
            else:
                data_dict[mode]['csv_path'] = csv_path
                #datum = {}
                #datum = process_map(convert_data_val, data_dict[mode]['imgs_path'],
                #                     max_workers=12)
                datum = []
                for path in tqdm(data_dict[mode]['imgs_path']):
                    datum.append(convert_data_val(path))
                df = pd.DataFrame(data=datum)
                df.to_csv(csv_path)


        df = pd.read_csv(data_dict['train']['csv_path'], index_col=0)
        print(df.head())
        df['classes'] = 0
        print(len(set(df['image_id'])))

    def f7(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]


    index = f7(df.image_id)


    random.Random(42).shuffle(index)
    print(len(index))

    val_index = ''
    if dataset_name == 'wheat':
        source = os.path.join(PATH, 'data', dataset_name, 'train')
    else:
        source = data_dict['train']['label_path'].split('label')[0]  # 'train'

    convertor_name = 'convertor'
    convertor_path = os.path.join(PATH, 'data', dataset_name, convertor_name)
    if not os.path.exists(convertor_path):
        os.mkdir(convertor_path)

    val_txt_path =  os.path.join(PATH, 'data', dataset_name, 'val.txt')
    train_path =  os.path.join(PATH, 'data', dataset_name, 'train.txt')
    for filepath in [val_txt_path, train_path]:
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                pass

    for fold in [0]:
        val_index = index[len(index)*fold//5:len(index)*(fold+1)//5]
        for name, mini in tqdm(df.groupby('image_id')):
            if dataset_name == 'wheat':
                sh.copy(os.path.join(source, 'images', f"{name}.jpg"),
                        os.path.join(convertor_path, f"{name}.jpg"))
            else:
                sh.copy(os.path.join(source, 'images', name),
                        os.path.join(convertor_path, f"{name.split('/')[-1].split('.')[0]}.jpg"))
            if name in val_index:
                path2save = val_txt_path  # f'{convertor_name}/val.txt'
            else:
                path2save = train_path  # f'{convertor_name}/train.txt'
            name = name.split('/')[-1].split('.')[0]
            with open(path2save, 'a') as f:
                f.write(f'{name}.jpg')
                row = mini[['x','y','x1','y1','classes']].astype(int).values
                # row = row/1024
                row = row.astype(str)
                for j in range(len(row)):
                    text = ' '+','.join(row[j])
                    f.write(text)
                f.write('\n')

    print(len(df['image_id'].unique()))

    # Check Dataset
    img_paths = []

    def check_image(img_name):
        img_name = img_name.split('/')[-1]
        img_path = os.path.join(convertor_path, img_name)
        img = cv2.imread(img_path)
        if ((img.shape[0] == 0) or (img.shape[1] == 0)):
            print('img.shape[0] == 0')
            img_paths.append(img_path)
        if img is None:
            print(f'img {img} is None!')
            img_paths.append(img_path)
            pass
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(e, img_path)
            img_paths.append(img_path)
            pass
    
    #for imagename in tqdm(df['image_id'].unique()):
    #    check_image(imagename)
    # process_map(check_image, , max_workers=12)
    #print()

    for filename in tqdm(os.listdir(convertor_path)):
      if filename.endswith('.jpg'):
        try:
          img = Image.open(os.path.join(convertor_path, filename)) # open the image file
          img.verify() # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
          print('Bad file:', filename) #

    # get_ipython().system('python3 train.py -l 0.01 -g 0 -classes 1 -dir convertor -pretrained yolov4coco/yolov4.conv.137.pth -optimizer sgd -iou-type giou -train_label_path convertor/train.txt')

    # !rm convertor/*
