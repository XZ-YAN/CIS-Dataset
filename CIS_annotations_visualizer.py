# coding=UTF-8
# Code Author: Xuzhong Yan

import io, os, glob, base64, json, cv2, random
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw

def annotation_visualizer(image_set):
    json_path = 'dataset/annotations/'+image_set+'.json'
    image_folder = image_set+'/'
    save_path = './annotations_visualization/'+image_set+'/'
    if not os.path.exists(save_path): os.makedirs(save_path)
    
    # load a COCO json
    with open(json_path, 'r') as data:
        all_json_data = json.load(data)
        category_dict = {}
        for c in all_json_data['categories']:
            category_dict[c['id']] = c['name']

        for i in tqdm(range(len(all_json_data['images']))):
            # load an image
            image = cv2.imread(image_folder + all_json_data['images'][i]['file_name'])
            image = Image.fromarray(np.uint8(image))
            image_id = all_json_data['images'][i]['id']

            # load annotations
            for j in all_json_data['annotations']:
                if j['image_id'] == image_id:
                    polygon_array = np.array(j['segmentation'][0]).reshape(int(len(j['segmentation'][0])/2),2)
                    polygon, x_list, y_list = [], [], []
                    for p in polygon_array:
                        polygon.append((p[0],p[1]))
                        x_list.append(int(p[0]))
                        y_list.append(int(p[1]))

                    # draw an instance annotation on the image
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 150)
                    ImageDraw.Draw(image,'RGBA').polygon(polygon, outline=1, fill=color)
                    image = np.array(image)

                    # draw corresponding category                    
                    for c in category_dict:
                        if j['category_id'] == c:
                            category_label = category_dict[c]
                    
                    x, y = int((max(x_list)+min(x_list))/2), int((max(y_list)+min(y_list))/2)
                    cv2.putText(image, category_label, (x-10, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                    # save the image with instance annotation and category
                    cv2.imwrite(save_path + all_json_data['images'][i]['file_name'], image)
                    image = Image.fromarray(np.uint8(image))

if __name__ == '__main__':
    annotation_visualizer('val') # 'train', 'test'
