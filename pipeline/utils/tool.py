import os
import cv2
import glob
import numpy as np


def xywh_to_xyxy(bbox_xywh,image_wh=(0,0)):
        x,y,w,h = bbox_xywh[0],bbox_xywh[1],bbox_xywh[2],bbox_xywh[3]
        x1 = max(int(x),0)
        x2 = min(int(x+w),image_wh[0]-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),image_wh[1]-1)
        return x1,y1,x2,y2


def mkdir_if_missing(sdir):
    if not os.path.isdir(sdir):
        os.makedirs(sdir)
