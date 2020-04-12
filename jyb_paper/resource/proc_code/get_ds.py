# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:40:18 2020

@author: Administrator
"""

import random
import pathlib
import IPython.display as display

root_path="E:\\jia_you_bing\\train"


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

# 元组被解压缩到映射函数的位置参数中
def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label

def get_local_datasets(path_dir):
    data_root = pathlib.Path(path_dir)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    
    random.shuffle(all_image_paths)
    
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]
    ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
    #最终数据集
    image_label_ds = ds.map(load_and_preprocess_from_path_label)






