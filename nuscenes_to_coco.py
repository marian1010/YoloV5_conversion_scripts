from importlib_metadata import version
from nuimages import NuImages
import json
from tqdm import tqdm
import os
import yaml

#---------------------------------- DEFINE CONSTANTS HERE -----------------------------------------
DATA_ROOT = os.path.dirname(os.path.realpath(__file__))

#nuimages has same image dimensions for all images
IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 900

'''class_map dict maps nuimages classes to other class names.
ex. human.pedestrian.adult and human.pedestrian.child both get mapped to "person"
vehicle.ego is removed from the data labels because it's string is "None"
'''
class_map = {
			'animal': 'animal',
			'human.pedestrian.adult': 'person',
			'human.pedestrian.child': 'person',
			'human.pedestrian.construction_worker': 'person',
			'human.pedestrian.personal_mobility': 'person',
			'human.pedestrian.police_officer': 'person',
			'human.pedestrian.stroller': 'person',
			'human.pedestrian.wheelchair': 'person',
			'movable_object.barrier': 'barrier',
			'movable_object.debris': 'debris',
			'movable_object.pushable_pullable': 'object_pushable_pullable',
			'movable_object.trafficcone': 'traffic cone',
			'static_object.bicycle_rack': 'bicycle rack',
			'vehicle.bicycle': 'bicycle',
			'vehicle.bus.bendy': 'bus',
			'vehicle.bus.rigid': 'bus',
			'vehicle.car': 'car',
			'vehicle.construction': 'construction vechicle',
			'vehicle.emergency.ambulance': 'truck',
			'vehicle.emergency.police': 'car',
			'vehicle.motorcycle': 'motorcycle',
			'vehicle.trailer': 'trailer',
			'vehicle.truck': 'truck',
			'vehicle.ego': 'None', #ignore ego vehicle
			}

#-------------------------------  AutoGenerate nuimages.yaml file -------------------------------------
#Create YAML file per section #1 of https://github.com/ultralytics/yolov5/issues/12
with open(os.path.join(DATA_ROOT,'nuimages.yaml'),'w') as file:
	classes = sorted(list(set(class_map.values())))
	classes = [c for c in classes if c!= 'None']
	print(classes)
	
	data = {'train': os.path.join(DATA_ROOT,'images','train'),
			'val': os.path.join(DATA_ROOT,'images','val'),
			'nc': len(classes),
			'names': classes,
			}

	yaml.dump(data, file, sort_keys = False)

#-------------------------------  Dump Images & Create Label files -------------------------------------
#Create files per section #2 of https://github.com/ultralytics/yolov5/issues/12

#for version in ['train','val']:
	#print(f'Processing {version} set...')
version = 'val'
	#make image and label output directories if they don't exist
	#os.makedirs(os.path.join(DATA_ROOT,'images',version),exist_ok = True)
os.makedirs(os.path.join(DATA_ROOT,'labels',version),exist_ok = True)

nuim = NuImages(dataroot=DATA_ROOT, version='v1.0-'+version, verbose=True, lazy=True)

	#create reverse class index dictionary: {animal -> 0, barrier -> 1, bicycle -> 2, ...}
class_index_map = {obj_class:classes.index(class_map[obj_class]) for obj_class in class_map if class_map[obj_class] != 'None'}




class_dic_aux = dict()
class_dic_aux.update({"animal":0})
class_dic_aux.update({"barrier":1})
class_dic_aux.update({"bicycle":2})
class_dic_aux.update({"bicycle rack":3})
class_dic_aux.update({"bus":4})
class_dic_aux.update({"car":5})
class_dic_aux.update({"construction vechicle":6})
class_dic_aux.update({"debris":7}) 
class_dic_aux.update({"motorcycle":8})
class_dic_aux.update({"object_pushable_pullable":9})
class_dic_aux.update({"person":10})
class_dic_aux.update({"traffic cone":11})
class_dic_aux.update({"trailer":12})
class_dic_aux.update({"truck":13})

class_dic = dict()
class_dic["type"] = "instances"
class_dic.update({"categories":[

{"id":0,"name":"animal","supercategory":"none"},    
{"id":1,"name":"barrier","supercategory":"none"},
{"id":2,"name":"bicycle","supercategory":"none"},
{"id":3,"name":"bicycle rack","supercategory":"none"},
{"id":4,"name":"bus","supercategory":"none"},
{"id":5,"name":"car","supercategory":"none"},
{"id":6,"name":"construction vechicle","supercategory":"none"},
{"id":7,"name":"debris","supercategory":"none"},
{"id":8,"name":"motorcycle","supercategory":"none"},
{"id":9,"name":"object_pushable_pullable","supercategory":"none"},
{"id":10,"name":"person","supercategory":"none"},
{"id":11,"name":"traffic cone","supercategory":"none"},
{"id":12,"name":"trailer","supercategory":"none"},
{"id":13,"name":"truck","supercategory":"none"}
]})

images = list()
position = list()
ignored_categoris = list()

counter= 0

for sample_idx in tqdm(range(0,len(nuim.sample))):
    sample = nuim.get('sample', nuim.sample[sample_idx]['token'])
    key_camera_token = sample['key_camera_token']

    object_tokens, surface_tokens = nuim.list_anns(sample['token'], verbose = False)

    #nuim.render_image(key_camera_token, annotation_type='none',with_category=True, with_attributes=True, box_line_width=-1, render_scale=5, 
                        #out_path = os.path.join(DATA_ROOT,'images',version,f'{sample_idx}.jpg'))		



    #with open(os.path.join(DATA_ROOT,'labels',version,f'{sample_idx}.txt'),'w') as file:
        
           
    counter += 1
    image = dict()
    image['id'] = counter   
    image['height'] = IMAGE_HEIGHT
    image['width'] = IMAGE_WIDTH
    image['file_name']= str(sample_idx)+'.txt'
    
    sin_imagen=True
    tmp = 0
    num_objet =0
    for object_token in object_tokens:
        token_data = nuim.get('object_ann',object_token)
        token_name = nuim.get('category',token_data['category_token'])['name']

        annotation = dict()

        if class_map[token_name] != 'None':
            sin_imagen=False
            tmp=1
            
            annotation['id'] = num_objet  ##revisar
            annotation["image_id"] = image['id']
            annotation['category_id'] = class_index_map[token_name]
                        
            x1=token_data['bbox'][0]
            x2=token_data['bbox'][2]
            y1=token_data['bbox'][1]
            y2=token_data['bbox'][3]   
            annotation['segmentation']=[[x1,y1,x1,y2,x2,y2,x2,y1]]             
            annotation['area'] = float((x2 - x1) * (y2 - y1))
            annotation['bbox'] = [x1, y1,x2 -x1, y2 - y1]
            annotation["iscrowd"] = 0  
            #annotation['ignore'] = 0
            
            
            position.append(annotation)             
            num_objet +=1
        #else:                
            #ignored_categoris.append(class_index_map[token_name])

    if sin_imagen:
        print('empty image!')
    if tmp == 1:
        images.append(image)

            

    class_dic["images"] = images
    class_dic["annotations"] = position
    class_dic["info"] = ""
    class_dic["licenses"] = ""

with open(DATA_ROOT+"nuscenes_annotation.json","a") as destfile:
    json.dump(class_dic,destfile,indent=4,sort_keys=True)
        
    
print ('Finish')

				#if class_map[token_name] != 'None':
					
                    #file.writelines('{} {} {} {} {}\n'.format(class_index_map[token_name],
															#(token_data['bbox'][2] + token_data['bbox'][0])/(IMAGE_WIDTH * 2),
															#(token_data['bbox'][3] + token_data['bbox'][1])/(IMAGE_HEIGHT * 2),
															#(token_data['bbox'][2] - token_data['bbox'][0])/IMAGE_WIDTH,
															#(token_data['bbox'][3] - token_data['bbox'][1])/IMAGE_HEIGHT,
															#))