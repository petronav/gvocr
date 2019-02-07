import os
import cv2
from PIL import Image
import pytesseract
import argparse
import imutils
import subprocess
import bs4
import difflib
import re
import math
import pandas as pd
import operator
from operator import itemgetter
from pprint import pprint
import scipy as sp
from hocr_parser.parser import HOCRDocument
import numpy as np
from scipy import ndimage

"""
input_filename = "./static/enhance_Alg_1.jpeg"
output_img_filename = ".".join(input_filename.split(".")[:-1]) + "_rot." + input_filename.split(".")[-1]
print(output_img_filename)
output_hocr_filename =  ".".join(output_img_filename.split(".")[:-1])
print(output_hocr_filename)
"""

def run_tesseract(input_file, output_file, lang=None):
    if lang is not None:
        command = ["tesseract", input_file, output_file, "-l", lang, "hocr"]
    else:
        command = ["tesseract", input_file, output_file, "-l eng hocr"]
    proc = subprocess.Popen(command, stderr = subprocess.PIPE)
    print("tesseract done.")
    return (proc.wait(), proc.stderr.read())
#run_tesseract(input_filename, output_hocr_filename, "eng")

def parse_hocr(hocr_file=None, regex=None):
    # Open the hocr file, read it into BeautifulSoup and extract all the ocr words.
    hocr = open(hocr_file,'r', encoding="utf-8").read()
    soup = bs4.BeautifulSoup(hocr,'html.parser')
    lines = soup.find_all('span',class_='ocr_line')
    angle = []
    for line in lines:
        line_title = line["title"].split(";")
        for i in line_title:
            if "baseline" in i:
                #print("i : ",i)
                angle.append(float(i.split()[-2]))
                
        rot_angle = max(angle) if len(angle) != 0 else 0
    #print(angle)
    if len(angle) > 0:
        avrangle = sum(angle)/len(angle)
    else:
        avrangle = 0
    #print(avrangle)
    rot_angle = math.degrees(math.atan(avrangle))
    print("rot_angle : ", rot_angle)
    return int(rot_angle)

def rotate_image(cv_image, angle,file_name_rot):
    image = cv2.imread(cv_image)
    rotated = None
    rotated_filename = ""
    if angle != 0:
        result = ndimage.rotate(image, angle)
        output_img = cv2.imwrite(file_name_rot, result)
        rotated = True
        print("rotated.")
    else:
        rotated = False
        print("not rotated.")
    return rotated
#check_rot = rotate_image(cv_image = input_filename, angle = parse_hocr(hocr_file = output_hocr_filename  + ".hocr"), file_name_rot = output_img_filename)
#print("check_rot : ", check_rot)