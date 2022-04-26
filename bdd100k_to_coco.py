import argparse
from cProfile import label
import json
from operator import ge
import os
import sys, getopt
from unicodedata import category
from tqdm import tqdm


def dbb100_to_yoloV5(file,outputDir):
    
    class_dic = dict()
    class_dic.update({"person":0})
    class_dic.update({"rider":1})
    class_dic.update({"car":2})
    class_dic.update({"truck":3})
    class_dic.update({"bus":4})
    class_dic.update({"train":5})
    class_dic.update({"motor":6}) 
    class_dic.update({"traffic light":7})
    class_dic.update({"traffic sign":8})

    images = list()
    position = list()
    ignored_categoris = list()

    counter= 0

    for i in tqdm(file):
        counter += 1
        image = dict()
        image['file_name']=i['name']
        image['height'] = 720
        image['width'] = 1280
        image['id'] = counter
         
        sin_imagen=True
        tmp = 0
        for j in i['labels']:
            annotation = dict()

            if j['category'] in class_dic.keys(): 
                sin_imagen=False
                tmp=1
                annotation["iscrowd"] = 0
                annotation["image_id"] = image['id']
                x1=j['box2d']['x1']
                x2=j['box2d']['x2']
                y1=j['box2d']['y1']
                y2=j['box2d']['y2']
                annotation['bbox'] = [x1, y1,x2 -x1, y2 - y1]
                annotation['area'] = float((x2 - x1) * (y2 - y1))
                annotation['category_id'] = class_dic[j['category']]
                annotation['ignore'] = 0
                annotation['id'] = j['id']
                annotation['segmentation']=[[x1,y1,x1,y2,x2,y2,x2,y1]]
                position.append(annotation)             
  
            else:                
                ignored_categoris.append(j['category'])
            if sin_imagen:
                print('empty image!')
            if tmp == 1:
                images.append(image)

        class_dic["images"] = images
        class_dic["position"] = images
        class_dic["type"] = "instances"

        with open(outputDir,"w") as destfile:
            json.dump(class_dic,destfile)
              
        
    print ('Finish')






def main(argv):
    inputfile = ''
    outputDir= ''
    named= ''
    options, args=getopt.getopt(argv,"hi:o:n",["infile=","outputDir=","nameFile"])

    if(len(options)==0):
        print ('-i imputfile -o outputfile')
        sys.exit(2)

    for i , arg in options:
        if(i == '-h'):
            print ('-i imputfile -o outputDir')
            sys.exit()
        elif i in ("-i","--infile"):
            inputfile = arg
        elif i in ("-o","--outputDir"):
            outputDir = arg    
        elif i in ("-n","--nameFile"):
            named = arg + ".json"

    print ('Input file is ', inputfile)
    print ('Output file is ', outputDir)
    print ('Name file is ', named)
    with open(inputfile) as file:
        labels =json.load(file)

    dbb100_to_yoloV5(labels,outputDir+named)

if __name__ == "__main__":
    main(sys.argv[1:])

