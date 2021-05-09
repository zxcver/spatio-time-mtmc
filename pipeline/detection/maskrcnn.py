"""
torch inference
"""
import os
import os.path as osp
import unittest

import torch
import argparse
import cv2
import numpy as np

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer


CFG_FILES = {
    'res50': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
    'res101': 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
    'res101x': 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml',
}

def drawsave_result(predict,image):
    #visual
    v = Visualizer(image[:, :, ::-1])
    v = v.draw_instance_predictions(predict["instances"].to("cpu"))
    cv2.imwrite('hehe1.jpg', v.get_image()[:, :, ::-1])


def analyse_outputs(predictions, image, writer, counter, nms_thres):
    if predictions.has("pred_boxes"):
        boxes = predictions.pred_boxes.tensor.numpy()
    else:
        boxes = None
    if predictions.has("pred_classes"):
        category = predictions.pred_classes.numpy()
    else:
        category = None
    if predictions.has("scores"):
        scores = predictions.scores.numpy()
    else:
        scores = None

    det_results = []
    for index in range(len(boxes)):
        if category[index] in [2,5,7]:
            det_result = boxes[index].tolist()
            det_result.append(scores[index])
            det_results.append(det_result)
    
    if det_results:
        det = np.array(det_results)
        nms_keep = py_xyxy_nms(det,nms_thres)

        nms_results = []
        for reserve_order in nms_keep:
            nms_results.append(det_results[reserve_order])

        # nonms_img = image.copy()
        # for det_result in det_results:
        #     x1, y1, x2, y2 = int(det_result[0]),int(det_result[1]), int(det_result[2]),int(det_result[3])
        #     cv2.rectangle(nonms_img, (x1, y1), (x2, y2), color=(0,255,0), thickness=2, lineType=8)
        #     cv2.imwrite('resultvisual/no-nms.jpg',nonms_img)
        
        # nms_img = image.copy()
        # for nms_result in nms_results:
        #     x1, y1, x2, y2 = int(nms_result[0]),int(nms_result[1]), int(nms_result[2]),int(nms_result[3])
        #     cv2.rectangle(nms_img, (x1, y1), (x2, y2), color=(0,255,0), thickness=2, lineType=8)
        #     cv2.imwrite('resultvisual/nms.jpg',nms_img)

        # 1,-1,1212.938,771.909,661.123,261.700,0.999,-1,-1,-1
        for vehicle_mes in nms_results:
            x1, y1, x2, y2, score = int(vehicle_mes[0]),int(vehicle_mes[1]),int(vehicle_mes[2]),int(vehicle_mes[3]),float(vehicle_mes[4])
            w,h = x2-x1, y2-y1
            line = "{:d},-1,{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},-1,-1,-1\n".format(counter,x1,y1,w,h,score)
            writer.writelines(line)


def py_xyxy_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def run(args):
    if torch.cuda.is_available():
        device = 'cuda:{}'.format(args.sgpu_infer) 
    cfg_file = CFG_FILES[args.default_model]

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.det_thres
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_file)
    cfg.MODEL.DEVICE = device

    cfg.INPUT.MIN_SIZE_TEST = args.min_size_test 
    cfg.INPUT.MAX_SIZE_TEST = args.max_size_test

    cfg.MODEL.DEVICE = device

    model_device = torch.device(cfg.MODEL.DEVICE)
    model_predictor = DefaultPredictor(cfg)

    scene_path = osp.join(args.input_root, args.scence_id)
    cams = os.listdir(scene_path)
    for cam in cams:
        video = os.path.join(scene_path, cam, 'vdo.avi')
        w_path = os.path.join(args.output_root,args.output_floder,cam,'det')
        if not os.path.exists(w_path):
            os.makedirs(w_path)
        w = open(os.path.join(w_path,args.output_name),'w')
        vc = cv2.VideoCapture(video)
        fps = vc.get(cv2.CAP_PROP_FPS)
        num_frame = vc.get(cv2.CAP_PROP_FRAME_COUNT)
        counter = 1
        while True:
            rval, frame = vc.read()
            if not rval:
                break
            predict = model_predictor(frame)
            # print(predict)
            # drawsave_result(predict,img)
            analyse_outputs(predict["instances"].to("cpu"), frame, w, counter, args.nms_thres)
            counter+=1
        w.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #gpu
    parser.add_argument("--sgpu_infer", type=str, default="0", help="single gpu for model infer")
    #path
    parser.add_argument('--input_root', type=str, default="datasets/AIC21_Track3_MTMC_Tracking/validation",help="path to the input image")
    parser.add_argument("--output_root", type=str, default="resultpipeline/det2", help="expected output root path")
    parser.add_argument('--scence_id', type=str, default="S02",help="scence id")
    parser.add_argument('--nms_thres', type=float, default=0.6,help="nms iou thres")
    parser.add_argument('--det_thres', type=float, default=0.4,help="det confidence thres")
    #result
    parser.add_argument("--default_model", type=str, default="res101x", help="expected backbone")
    parser.add_argument("--output_name", type=str, default="mask_rcnn_X_101_32x8d_FPN_3x.txt", help="expected output name") 
    args = parser.parse_args()
    run(args)
