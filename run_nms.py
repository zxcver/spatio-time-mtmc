import os
import os.path as osp
import numpy as np 
from evaluation.evaldet.load import FRestructMot
import argparse


def py_xywh_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    w = dets[:, 2]
    h = dets[:, 3]
    x2 = x1+w
    y2 = y1+h
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


def det4frame(detfile_path):
    with open(detfile_path, "r") as f:
        motgt = f.readlines()
    valid_redict = FRestructMot(motgt)
    return valid_redict

def main(args):
    sorted_cams = ['c041','c042','c043','c044','c045','c046']
    for cam in sorted_cams:
        detfile_path_800 = osp.join(args.det_path_800,cam,'det','mask_rcnn_X_n6_101_32x8d_FPN_3x.txt')
        valid_redict_800 = det4frame(detfile_path_800)
        detfile_path_1200 = osp.join(args.det_path_1200,cam,'det','mask_rcnn_X_n6_101_32x8d_FPN_3x.txt')
        valid_redict_1200 = det4frame(detfile_path_1200)
        detfile_path_1600 = osp.join(args.det_path_1600,cam,'det','mask_rcnn_X_n6_101_32x8d_FPN_3x.txt')
        valid_redict_1600 = det4frame(detfile_path_1600)

        write_path = osp.join(args.out_path,cam,'det')
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        wtiter = open(osp.join(write_path,'mask_rcnn_X_n6_101_32x8d_FPN_3x.txt'),'w')

        for i in range(1,2002):
            if i in valid_redict_800.keys():
                frame_800 = valid_redict_800[i]
            if i in valid_redict_1200.keys():
                frame_1200 = valid_redict_1200[i]
            if i in valid_redict_1600.keys():
                frame_1600 = valid_redict_1600[i]

            det_results = []
            for instance in frame_800:
                bbox,_,score = instance
                det_result = list(bbox)
                det_result.append(score)
                det_results.append(det_result)
            
            for instance in frame_1200:
                bbox,_,score = instance
                det_result = list(bbox)
                det_result.append(score)
                det_results.append(det_result)

            for instance in frame_1600:
                bbox,_,score = instance
                det_result = list(bbox)
                det_result.append(score)
                det_results.append(det_result)
            
            if det_results:
                det = np.array(det_results)
                nms_keep = py_xywh_nms(det,0.6)

            nms_results = []
            for reserve_order in nms_keep:
                nms_results.append(det_results[reserve_order])

            for nms_result in nms_results:
                x1, y1, w, h, score = int(nms_result[0]),int(nms_result[1]),int(nms_result[2]),int(nms_result[3]),float(nms_result[4])
                line = "{:d},-1,{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},-1,-1,-1\n".format(i,x1,y1,w,h,score)
                wtiter.writelines(line)
        wtiter.close()


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_path_800', type=str, default="allin/det/S06-800",help="800 scale det result")
    parser.add_argument("--det_path_1200", type=str, default="allin/det/S06-1200", help="1200 scale det result")
    parser.add_argument('--det_path_1600', type=str, default="allin/det/S06-1600",help="1600 scale det result")
    parser.add_argument('--out_path', type=str, default="allin/det/S06",help="out path")
    parser.add_argument('--nms_thres', type=float, default=0.6,help="nms iou thres")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argument_parser()
    main(args)