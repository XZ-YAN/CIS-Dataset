# coding=UTF-8
# Code Author: Xuzhong Yan

import os, sys, time, glob, math, cv2, turtle, ast, matplotlib, random, json, seaborn
from tqdm import tqdm
from functools import reduce
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import matplotlib.pyplot as plt
from matplotlib import pyplot, transforms, cm
from matplotlib.ticker import ScalarFormatter
from ast import literal_eval
from copy import deepcopy
from shapely.geometry import box, Polygon
from imantics import Mask
import numpy as np
import numpy.random
from io import StringIO
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy.optimize import fsolve
import mpl_toolkits.mplot3d.art3d as art3d
from scipy.stats import multivariate_normal, gaussian_kde
from scipy import stats
from matplotlib.ticker import LinearLocator
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import matplotlib.colors as cor
from collections import Counter

def total_number_of_instances(json_file_path): # calculate total number of imgaes and instances
    with open(json_file_path, 'r') as data:
        json_data = json.load(data)
    print('-----Number of images: ', len(json_data['images']))
    print('\n-----Number of instances: ', len(json_data['annotations']))

def number_of_instances_and_images_per_category(json_file_path, save_path): # calculate total number of imgaes and instances in each object category
    with open(json_file_path, 'r') as data:
        json_data = json.load(data)

    number_of_instances_per_category = {}
    number_of_images_per_category = {}
    categories = []
    for category in json_data['categories']:
        categories.append(category['name'])
        number_of_instances_per_category[category['name']] = 0
        number_of_images_per_category[category['name']] = 0   

    # calculate total number of instances in each object category
    for annotation in json_data['annotations']: 
        for i in range(10):
            if annotation['category_id'] == i:
                number_of_instances_per_category[categories[i]] += 1
    # calculate total number of imgaes in each object category
    previous_image_id = []
    for annotation in json_data['annotations']:
        if annotation['image_id'] not in previous_image_id:
            previous_image_id.append(annotation['image_id'])
            previous_category_id = [annotation['category_id']]
            for i in range(10):
                if annotation['category_id'] == i:
                    number_of_images_per_category[categories[i]] += 1
            
        else:
            if annotation['category_id'] not in previous_category_id:
                
                previous_category_id.append(annotation['category_id'])
                for i in range(10):
                    if annotation['category_id'] == i:
                        number_of_images_per_category[categories[i]] += 1

    # reorder keys
    desired_order = ['people-helmet', 'people-no-helmet', 'PC', 'PC-truck', 'dump-truck', 'mixer', 'excavator', 'roller', 'dozer', 'wheel-loader']
    number_of_instances_per_category = {k: number_of_instances_per_category[k] for k in desired_order}
    number_of_images_per_category = {k: number_of_images_per_category[k] for k in desired_order}

    # plot
    fig, ax = plt.subplots(figsize=(12,6), dpi=1000)
    plt.rcParams['font.size'] = '22'
    plt.rcParams['font.family'] = 'Times New Roman'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(22)
  
    x_categories = list(number_of_instances_per_category.keys())
    x_axis = np.arange(len(number_of_instances_per_category))
    y_1 = list(number_of_instances_per_category.values())
    y_2 = list(number_of_images_per_category.values())

    ax.bar(x_axis-0.2, y_1, 0.4, label='Instances')
    ax.bar(x_axis+0.2, y_2, 0.4, label='Images')
    plt.xticks(range(len(number_of_images_per_category)), list(number_of_images_per_category.keys()), rotation=30, ha='right', font='Times New Roman')
    plt.yticks(font='Times New Roman')
    ax.legend()
    plt.yscale('log', base=10)
    ax.set_yticks([10, 100, 1000, 10000, 100000])
    plt.savefig(save_path + '1-Instances-Images-per-Category.jpg', bbox_inches='tight')

def number_of_instances_per_category_on_train_val_test(train_json_path, val_json_path, test_json_path, save_path): # calculate total number of imgaes and instances in train, val, and test set
    with open(train_json_path, 'r') as data: train_json_data = json.load(data)
    with open(val_json_path, 'r') as data: val_json_data = json.load(data)
    with open(test_json_path, 'r') as data: test_json_data = json.load(data)

    number_of_instances_per_category_train = {}
    number_of_instances_per_category_val = {}
    number_of_instances_per_category_test = {}
    categories = []
    for category in train_json_data['categories']:
        categories.append(category['name'])
        number_of_instances_per_category_train[category['name']] = 0
        number_of_instances_per_category_val[category['name']] = 0
        number_of_instances_per_category_test[category['name']] = 0
        
    # calculate total number of instances in each object category
    for annotation in train_json_data['annotations']: 
        for i in range(10):
            if annotation['category_id'] == i:
                number_of_instances_per_category_train[categories[i]] += 1
    for annotation in val_json_data['annotations']: 
        for i in range(10):
            if annotation['category_id'] == i:
                number_of_instances_per_category_val[categories[i]] += 1
    for annotation in test_json_data['annotations']: 
        for i in range(10):
            if annotation['category_id'] == i:
                number_of_instances_per_category_test[categories[i]] += 1

    # reorder keys
    desired_order = ['people-helmet', 'people-no-helmet', 'PC', 'PC-truck', 'dump-truck', 'mixer', 'excavator', 'roller', 'dozer', 'wheel-loader']
    number_of_instances_per_category_train = {k: number_of_instances_per_category_train[k] for k in desired_order}
    number_of_instances_per_category_val = {k: number_of_instances_per_category_val[k] for k in desired_order}
    number_of_instances_per_category_test = {k: number_of_instances_per_category_test[k] for k in desired_order}
    
    # plot
    fig, ax = plt.subplots(figsize=(12,6), dpi=1000)
    plt.rcParams['font.size'] = '22'
    plt.rcParams['font.family'] = 'Times New Roman'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(22)
  
    x_categories = list(number_of_instances_per_category_train.keys())
    x_axis = np.arange(len(number_of_instances_per_category_train))
    y_1 = list(number_of_instances_per_category_train.values())
    y_2 = list(number_of_instances_per_category_val.values())
    y_3 = list(number_of_instances_per_category_test.values())

    ax.bar(x_axis-0.2, y_1, 0.2, label='Train')
    ax.bar(x_axis,     y_2, 0.2, label='Validation')
    ax.bar(x_axis+0.2, y_3, 0.2, label='Test')
    plt.xticks(range(len(number_of_instances_per_category_train)), list(number_of_instances_per_category_train.keys()), rotation=30, ha='right', font='Times New Roman')
    plt.yticks(font='Times New Roman')
    ax.legend()
    plt.yscale('log', base=10)
    ax.set_yticks([1, 10, 100, 1000, 10000, 100000])
    plt.savefig(save_path + '2-Instances-per-Category-in-Train-Val-Test.jpg', bbox_inches='tight')

def number_of_instances_and_categories_per_image(json_file_path, save_path): # calculate number of instances and categories in each image
    with open(json_file_path, 'r') as data: json_data = json.load(data)
    number_of_instances = []
    instances_per_image = {}
    categories_per_image = {}
    for i in range(20): number_of_instances.append(str(i+1))
    for i in number_of_instances:
        instances_per_image[i] = 0
        categories_per_image[i] = 0
    instances_per_image['>20'] = 0
    categories_per_image['>20'] = 0

    # calculate total number of instances in each image
    for i in range(len(json_data['images'])):
        instance_number = 0
        for annotation in json_data['annotations']:
            if annotation['image_id'] == i:
                instance_number += 1

        if instance_number<=20:
            instances_per_image[str(instance_number)]+=1
        else:
            instances_per_image['>20']+=1
    s = sum(instances_per_image.values())
    for i in range(20):
        instances_per_image[str(i+1)] = instances_per_image[str(i+1)]/s
    instances_per_image['>20'] = instances_per_image['>20']/s

    # calculate total number of categories in each image
    for i in range(len(json_data['images'])):
        category_list = []
        for annotation in json_data['annotations']:
            if annotation['image_id'] == i and annotation['category_id'] not in category_list:
                category_list.append(annotation['category_id'])

        if len(category_list)<=20:
            categories_per_image[str(len(category_list))]+=1
        else:
            categories_per_image['>20']+=1
    s = sum(categories_per_image.values())
    for i in categories_per_image:
        categories_per_image[i] = categories_per_image[i]/s

    # plot
    fig, ax = plt.subplots(figsize=(12,6), dpi=1000)
    plt.rcParams['font.size'] = '22'
    plt.rcParams['font.family'] = 'Times New Roman'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(22)
  
    x_categories = list(instances_per_image.keys())
    x_axis = np.arange(len(instances_per_image))
    y_1 = list(instances_per_image.values())
    y_2 = list(categories_per_image.values())

    ax.plot(x_categories, y_1, marker='.', markersize=15, label='Instances')
    ax.plot(x_categories, y_2, marker='^', markersize=10, label='Categories')
    plt.xticks(font='Times New Roman')
    plt.yticks(font='Times New Roman')
    ax.set_xlabel('Number of Instances/Categories', fontsize=22, font='Times New Roman')
    ax.set_ylabel('Percentage of Images', fontsize=22, font='Times New Roman')
    ax.legend()

    for l in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        ax.axhline(y=l, linewidth=0.1, color='gray')
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.savefig(save_path + '3-Instances-Categories-per-Image.jpg', bbox_inches='tight')

def percentage_of_image_size(json_file_path, save_path): # calculate the percentage of mask area in each image
    with open(json_file_path, 'r') as data: json_data = json.load(data)
    percentage_of_size = {'(0%, 10%]':0,'(10%, 20%]':0,'(20%, 40%]':0,'(40%, 60%]':0,'(60%, 100%]':0}

    for i in range(len(json_data['images'])):
        image_area = json_data['images'][i]['height']*json_data['images'][i]['width']
        for annotation in json_data['annotations']:
            if annotation['image_id'] == i:
                area_ratio = annotation['area']/image_area
                if area_ratio>0 and area_ratio<=0.1: percentage_of_size['(0%, 10%]']+=1
                elif area_ratio>0.1 and area_ratio<=0.2: percentage_of_size['(10%, 20%]']+=1
                elif area_ratio>0.2 and area_ratio<=0.4: percentage_of_size['(20%, 40%]']+=1
                elif area_ratio>0.4 and area_ratio<=0.6: percentage_of_size['(40%, 60%]']+=1
                elif area_ratio>0.6 and area_ratio<=1: percentage_of_size['(60%, 100%]']+=1

    s = sum(percentage_of_size.values())
    for i in percentage_of_size:
        percentage_of_size[i] = percentage_of_size[i]/s

    # plot
    fig, ax = plt.subplots(figsize=(12,6), dpi=1000)
    plt.rcParams['font.size'] = '22'
    plt.rcParams['font.family'] = 'Times New Roman'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(22)
  
    x_categories = list(percentage_of_size.keys())
    x_axis = np.arange(len(percentage_of_size))
    y = list(percentage_of_size.values())
    ax.bar(x_categories, y, 0.4)
    plt.xticks(font='Times New Roman')
    plt.yticks(font='Times New Roman')
    ax.set_xlabel('Percentage of Image Size', fontsize=22, font='Times New Roman')
    ax.set_ylabel('Percentage of Instances', fontsize=22, font='Times New Roman')

    for l in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        ax.axhline(y=l, linewidth=0.1, color='gray')
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.savefig(save_path + '4-Percentage-of-Size.jpg', bbox_inches='tight')

def image_size_distribution(json_file_path, save_path): # visualize image size distribution
    with open(json_file_path, 'r') as data: json_data = json.load(data)
    all_width = []
    all_height = []
    all_size = []
    for i in range(len(json_data['images'])):
        width = json_data['images'][i]['width']
        height = json_data['images'][i]['height']
        all_width.append(width)
        all_height.append(height)
        all_size.append([width, height])

    counts = Counter(map(tuple, all_size)) # count number of each image size
    
    max_range = 10000
    bin_size = 10
    bin_num = int(max_range/bin_size)
    
    x = np.linspace(0, 10000, bin_size)
    y = np.linspace(0, 10000, bin_size)
    xx, yy = np.meshgrid(x, y)    
    z = xx*0+yy*0+ np.zeros((bin_size,bin_size))
    
    for size in all_size:
        for i in range(bin_size):
            for j in range(bin_size):
                if size[0]>bin_num*i and size[0]<bin_num*(i+1) and size[1]>bin_num*j and size[1]<bin_num*(j+1):
                    z[i][j] += 1

    # Max/Min Normalization of z
    for r in range(bin_size):
        for c in range(bin_size):
            z[r][c] = (z[r][c]-np.amin(z))/(np.amax(z)-np.amin(z))

    # plot
    fig, ax = plt.subplots(figsize=(8,6), dpi=1000)
    plt.rcParams['font.size'] = '22'
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.xticks(font='Times New Roman')
    plt.yticks(font='Times New Roman')
    ax.set_xlabel('Image Width', fontsize=22, font='Times New Roman')
    ax.set_ylabel('Image Height', fontsize=22, font='Times New Roman')
    
    plt.imshow(z, interpolation='nearest', cmap=plt.cm.viridis, extent=[0,10000,0,10000])
    plt.colorbar()
    plt.savefig(save_path + '5-Image-Size-Distribution.jpg', bbox_inches='tight')

def image_number_vs_instance_number(json_file_path, save_path): # image number v.s. instance number
    with open(json_file_path, 'r') as data:
        json_data = json.load(data)

    # Image number and instance number of different dataset
    CIS = [50000, int(len(json_data['annotations']))]
    COCO = [200000, 500000]
    PASCAL = [11530, 6929]
    Caltech_256 = [30607, 29780]
    Caltech_101 = [9146, 9144]
    Cityscapes = [2975+500+1525, 25000]

    # plot
    fig, ax = plt.subplots(figsize=(5,3), dpi=600)
    plt.rcParams['font.size'] = '14'
    plt.rcParams['font.family'] = 'Times New Roman'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(22)
    # CIS
    ax.scatter(CIS[0],CIS[1], c='red', s=200)
    plt.text(CIS[0]*1.2,CIS[1]*1.2, 'CIS', fontsize = 10)
    # COCO
    ax.scatter(COCO[0],COCO[1], c='green', s=200)
    plt.text(COCO[0]*1.2,COCO[1]*1.2, 'COCO', fontsize = 10)
    # PASCAL
    ax.scatter(PASCAL[0],PASCAL[1], c='blue', s=200)
    plt.text(PASCAL[0]*1.2,PASCAL[1]*0.8, 'PASCAL', fontsize = 10)
    # Cityscapes
    ax.scatter(Cityscapes[0],Cityscapes[1], c='pink', s=200)
    plt.text(Cityscapes[0]*1.2,Cityscapes[1]*1.2, 'Cityscapes', fontsize = 10)
    # Caltech-256
    ax.scatter(Caltech_256[0],Caltech_256[1], c='purple', s=200)
    plt.text(Caltech_256[0]*1.2,Caltech_256[1]*1.2, 'Caltech-256', fontsize = 10)
    # Caltech-101
    ax.scatter(Caltech_101[0],Caltech_101[1], c='orange', s=200)
    plt.text(Caltech_101[0]*1.2,Caltech_101[1]*1.2, 'Caltech-101', fontsize = 10)


    plt.xticks(font='Times New Roman')
    plt.yticks(font='Times New Roman')
    ax.set_xlabel('Number of Images', fontsize=10, font='Times New Roman')
    ax.set_ylabel('Number of Instance Masks', fontsize=10, font='Times New Roman')

    plt.xscale('log', base=10)
    plt.yscale('log', base=10)
    ax.set_xticks([1000, 10000, 100000, 1000000])
    ax.set_yticks([1000, 10000, 100000, 1000000])

    for l in [1000, 10000, 100000, 1000000]:
        ax.axvline(x=l, linewidth=0.3, color='gray')
        ax.axhline(y=l, linewidth=0.3, color='gray')
    plt.savefig(save_path + '6-Image-VS-Instance.jpg', bbox_inches='tight')

def all_location_distribution(json_file_path, save_path): # visualize mask center distribution - all categories
    with open(json_file_path, 'r') as data: json_data = json.load(data)
    center_coordinates_x = []
    center_coordinates_y = []
    for i in range(len(json_data['images'])):
        for annotation in json_data['annotations']:
            if annotation['image_id'] == i:
                # center x and center y
                center_x = annotation['bbox'][0]+annotation['bbox'][2]/2
                center_y = annotation['bbox'][1]+annotation['bbox'][3]/2
                # normalize center x and center y
                norm_center_x = center_x / json_data['images'][i]['width']
                norm_center_y = center_y / json_data['images'][i]['height']
                center_coordinates_x.append(norm_center_x)
                center_coordinates_y.append(norm_center_y)
            
    # plot
    fig, ax = plt.subplots(figsize=(6,6), dpi=1000)
    plt.rcParams['font.family'] = 'Times New Roman'
    gs = GridSpec(6,6)

    ax_heatmap = fig.add_subplot(gs[1:6,0:5])
    ax_distribution_x = fig.add_subplot(gs[0:1,0:5])
    ax_distribution_y = fig.add_subplot(gs[1:6,5:6])
    
    # plot ax_heatmap
    sigma = 16 # 32 or 64
    bins = 100
    heatmap, xedges, yedges = np.histogram2d(center_coordinates_x,center_coordinates_y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma)
    ax_heatmap.imshow(heatmap, origin='upper', cmap=cm.jet) # 'upper' indicates (0,0) at the left upper corner
    ax_heatmap.get_xaxis().set_ticks([])
    ax_heatmap.get_yaxis().set_ticks([])

    # plot x distribution
    array_x = list(np.array(center_coordinates_x)*bins)
    density_x = gaussian_kde(array_x)
    density_x.covariance_factor = lambda : .25
    density_x._compute_covariance()
    x = np.arange(0, bins, .1)
    ax_distribution_x.plot(x, density_x(x), 'black', linewidth=1, label='Kernel Density Estimation')   
    ax_distribution_x.get_xaxis().set_ticks([])
    ax_distribution_x.set_yticks([0, 0.05])
    for l in [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
        ax_distribution_x.axhline(y=l, linewidth=0.1, color='gray')
    ax_distribution_x.legend(fontsize='12', loc='best')
    ax_distribution_x.tick_params(axis='both', labelsize=16)
    
    # plot y distribution
    array_y = list(np.array(center_coordinates_y)*bins)
    density_y = gaussian_kde(array_y)
    density_y.covariance_factor = lambda : .25
    density_y._compute_covariance()
    y = np.arange(0, bins, .1)
    rot, base = transforms.Affine2D().rotate_deg(-90), pyplot.gca().transData
    ax_distribution_y.plot(y, density_y(y), 'black', linewidth=1, label='Kernel Density Estimation', transform= rot + base)
    ax_distribution_y.get_yaxis().set_ticks([])
    ax_distribution_y.set_xticks([0, 0.05])
    for l in [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
        ax_distribution_y.axvline(x=l, linewidth=0.1, color='gray')
    ax_distribution_y.tick_params(axis='both', labelsize=16)

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.axis('off')
    plt.savefig(save_path + '7-Location-Distribution.jpg', bbox_inches='tight')
    
def location_distribution_of_each_category(json_file_path, save_path, category_id): # # visualize mask center distribution - each category
    if category_id == 0: category_name = 'PC'
    elif category_id == 1: category_name = 'PC-truck'
    elif category_id == 2: category_name = 'dozer'
    elif category_id == 3: category_name = 'dump-truck'
    elif category_id == 4: category_name = 'excavator'
    elif category_id == 5: category_name = 'mixer'
    elif category_id == 6: category_name = 'people-helmet'
    elif category_id == 7: category_name = 'people-no-helmet'
    elif category_id == 8: category_name = 'roller'
    elif category_id == 9: category_name = 'wheel-loader'
       
    with open(json_file_path, 'r') as data: json_data = json.load(data)
    center_coordinates_x = []
    center_coordinates_y = []
    for i in range(len(json_data['images'])):
        for annotation in json_data['annotations']:
            if annotation['image_id'] == i and annotation['category_id'] == category_id:
                
                # center x and center y
                center_x = annotation['bbox'][0]+annotation['bbox'][2]/2
                center_y = annotation['bbox'][1]+annotation['bbox'][3]/2
                # normalize center x and center y
                norm_center_x = center_x / json_data['images'][i]['width']
                norm_center_y = center_y / json_data['images'][i]['height']
                center_coordinates_x.append(norm_center_x)
                center_coordinates_y.append(norm_center_y)
            
    # plot
    fig, ax = plt.subplots(figsize=(6,6), dpi=1000)
    plt.rcParams['font.family'] = 'Times New Roman'
    gs = GridSpec(6,6)

    ax_heatmap = fig.add_subplot(gs[1:6,0:5])
    ax_distribution_x = fig.add_subplot(gs[0:1,0:5])
    ax_distribution_y = fig.add_subplot(gs[1:6,5:6])
    
    # plot ax_heatmap
    sigma = 16 # 32 or 64
    bins = 100
    heatmap, xedges, yedges = np.histogram2d(center_coordinates_x,center_coordinates_y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma)
    ax_heatmap.imshow(heatmap, origin='upper', cmap=cm.jet) # 'upper' indicates (0,0) at the left upper corner
    ax_heatmap.get_xaxis().set_ticks([])
    ax_heatmap.get_yaxis().set_ticks([])

    # plot x distribution
    array_x = list(np.array(center_coordinates_x)*bins)
    density_x = gaussian_kde(array_x)
    density_x.covariance_factor = lambda : .25
    density_x._compute_covariance()
    x = np.arange(0, bins, .1)
    ax_distribution_x.plot(x, density_x(x), 'black', linewidth=1)
    ax_distribution_x.get_xaxis().set_ticks([])
    ax_distribution_x.set_yticks([])
    
    # plot y distribution
    array_y = list(np.array(center_coordinates_y)*bins)
    density_y = gaussian_kde(array_y)
    density_y.covariance_factor = lambda : .25
    density_y._compute_covariance()
    y = np.arange(0, bins, .1)
    rot, base = transforms.Affine2D().rotate_deg(-90), pyplot.gca().transData
    ax_distribution_y.plot(y, density_y(y), 'black', linewidth=1, transform= rot + base)
    ax_distribution_y.get_yaxis().set_ticks([])
    ax_distribution_y.set_xticks([])

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.axis('off')
    plt.savefig(save_path + '7.'+str(category_id+1)+'-'+category_name+'.jpg', bbox_inches='tight')

if __name__ == '__main__':    
    all_json_file_path = './dataset/annotations/ALL-50000.json'
    train_json_file_path= './dataset/annotations/train.json'
    val_json_file_path= './dataset/annotations/val.json'
    test_json_file_path= './dataset/annotations/test.json'
    save_path = './statistics_visualization/'
    if not os.path.exists(save_path): os.makedirs(save_path)
    
    total_number_of_instances(all_json_file_path)
    number_of_instances_and_images_per_category(all_json_file_path, save_path)
    number_of_instances_per_category_on_train_val_test(train_json_file_path, val_json_file_path, test_json_file_path, save_path)
    number_of_instances_and_categories_per_image(all_json_file_path, save_path)
    percentage_of_image_size(all_json_file_path, save_path)
    image_size_distribution(all_json_file_path, save_path)
    image_number_vs_instance_number(all_json_file_path, save_path)
    
    all_location_distribution(all_json_file_path, save_path)
    for category_id in range(10):
        location_distribution_of_each_category(all_json_file_path, save_path, category_id)
