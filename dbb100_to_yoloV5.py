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
    class_dic.update({"pedestrian":0})
    class_dic.update({"rider":1})
    class_dic.update({"car":2})
    class_dic.update({"truck":3})
    class_dic.update({"bus":4})
    class_dic.update({"train":5})
    class_dic.update({"motorcycle":6})
    class_dic.update({"bicycle":7})
    class_dic.update({"traffic light":8})
    class_dic.update({"traffic sign":9})

    ##images = list()
    position = list()
    ignored_categoris = list()

    for i in tqdm(file):
        images = i['name']
        images = images[:-4]
        for j in i['labels']:

            if j['category'] in class_dic.keys(): 
                category = class_dic[j['category']]
                x1=j['box2d']['x1']
                x2=j['box2d']['x2']
                y1=j['box2d']['y1']
                y2=j['box2d']['y2']
                center_point_x = (x2+x1)/2
                center_point_y = (y2+y1)/2
                width = (x2-x1)
                height= (y2-y1)
                
                center_point_x /= 1280
                center_point_y /= 720 
                width          /= 1280
                height         /= 720 

                file = open(outputDir+images+".txt","a") 
                text =str(category)+" "+str(center_point_x)+" "+str(center_point_y)+" "+str(width)+" "+str(height)+"\n"
                file.write(text)
            else:
                
                ignored_categoris.append(j['category'])

        
    print ('Finish')






def main(argv):
    inputfile = ''
    outputDir= ''

    options, args=getopt.getopt(argv,"hi:o:",["infile=","outputDir="])

    if(len(options)==0):
        print ('coco_to_darknet.py -i imputfile -o outputfile')
        sys.exit(2)

    for i , arg in options:
        if(i == '-h'):
            print ('coco_to_darknet.py -i imputfile -o outputDir')
            sys.exit()
        elif i in ("-i","--infile"):
            inputfile = arg
        elif i in ("-o","--outputDir"):
            outputDir = arg    

    print ('Input file is ', inputfile)
    print ('Output file is ', outputDir)

    with open(inputfile) as file:
        labels =json.load(file)

    dbb100_to_yoloV5(labels,outputDir)

if __name__ == "__main__":
    main(sys.argv[1:])

