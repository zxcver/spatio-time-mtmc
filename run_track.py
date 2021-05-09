
"""
run multi-video track 
"""
# base
import os
import os.path as osp
import cv2
import time
import copy
import argparse
import numpy as np

# utils
from evaluation.evaldet import LoadMOTGT,FRestructMot,FusionIGA
from pipeline.utils.load import  load_multiaicity

from pipeline.utils.tool import mkdir_if_missing,xywh_to_xyxy
from pipeline.utils.summary import save_singleimg,write_allres
from pipeline.utils.summary import release_writer,release_matchmes,release_trackres
from pipeline.utils.summary import init_writer,init_matchmes,init_trackres,print_summary

from pipeline.normalnet import NormalnetEmb
from pipeline.fastnet import FastEmb
from pipeline.aicitynet import AicityEmb
from pipeline.densetrack import DenseTrack


def init_extractor(emb_type,emb_model):
    #init extractor
    if emb_type == 'normalnet':
        extractor = NormalnetEmb(emb_model)
    elif emb_type == 'aicitynet':
        extractor = AicityEmb(emb_model)
    elif emb_type == 'fastnet':
        extractor = FastEmb(emb_model,'configs/inference/config.yaml')
    else:
        print("error embedding type")
        return
    return extractor


# feat_alpha 用于前后帧特征比例平滑
# embedding_thre 特征匹配的阈值
# iou_thre1 第一次iou匹配阈值
# iou_thre2 第二次iou匹配阈值
def init_tracktor(track_type,allsub_imgs,conf,feat_alpha,embedding_thre,iou_thre1,iou_thre2):
    if track_type == 'dense':
        tracktor = DenseTrack(allsub_imgs, conf, feat_alpha,embedding_thre,iou_thre1,iou_thre2,frame_rate=10)  
    else:
        print("error track type")
        return
    return tracktor


def load_detresult(file_path,scence_id, cam_id,file_type,pro_len):
    all_DetGT = {}
    gtfiles = os.listdir(osp.join(file_path,scence_id,cam_id,'expand'))
    if len(gtfiles) < 1:
        print("no det offline file error")
        return
    gtfile = [i for i in gtfiles if file_type in i] 
    readfile = osp.join(osp.join(file_path,scence_id, cam_id,'expand'), gtfile[0])
    motgt = LoadMOTGT(readfile)  
    valid_redict = FRestructMot(motgt)
    all_DetGT[cam_id] = FusionIGA(pro_len,valid_redict)
    return all_DetGT


def eval_multiseq(args, allsub_imgs, pro_len):
    det_results = load_detresult(args.det_result, args.scence_id, args.cam_id, args.det_type, pro_len)
    extractor = init_extractor(args.emb_type,args.emb_model)
    tracktor = init_tracktor(args.track_type, allsub_imgs, args.det_conf, args.feat_alpha,args.embedding_thre, args.iou_thre1, args.iou_thre2)
    output_path = osp.join(args.output_root,args.scence_id,args.cam_id)
    if args.draw_save:
        writer_dict = init_writer(output_path,args.output_name,allsub_imgs)

    matchmes_dict = init_matchmes(allsub_imgs)
    trackres_dict = init_trackres(allsub_imgs)

    for frame_index in range(pro_len):
        if args.scence_id == 'S06' and frame_index+1 > 2000:
            continue 
        for seqkey in allsub_imgs.keys():
            if len(allsub_imgs[seqkey]) <= frame_index:
                print("index out range!")
                continue
            image_path = allsub_imgs[seqkey][frame_index]
            if not osp.isfile(image_path):
                print("no file!")
                continue
            matchmes_dict[seqkey]["frame_id"] += 1 
            start_det = time.time()
            
            # replace
            # det_list =  detector.detect(image_path)
            det_list = det_results[seqkey][frame_index+1]

            end_det = time.time()
            matchmes_dict[seqkey]["det_time"] += end_det - start_det
            start_reid = time.time()
            ori_img = cv2.imread(image_path)
            croped_imgs = []
            bbox_list = []
            score_list = []
            for det_sem in det_list:
                box_xywh = det_sem[0]
                score = det_sem[3]
                if score < args.det_conf:
                    continue
                if box_xywh[2]<args.det_size_w or box_xywh[3]<args.det_size_h:
                    continue
                if float(box_xywh[3]/box_xywh[2])>args.scale_hw:
                    continue
                bbox_list.append(box_xywh)
                score_list.append(score)
                x1,y1,x2,y2 = xywh_to_xyxy(box_xywh,(ori_img.shape[1],ori_img.shape[0]))
                # copy memory
                croped_im = ori_img[y1:y2,x1:x2]
                # cv2.imwrite("results/1.jpg",im)
                croped_imgs.append(croped_im)

            if len(croped_imgs) > 0:
                #input img need 4-dims
                id_feature = extractor.extract(croped_imgs)
            end_reid = time.time()
            matchmes_dict[seqkey]["reid_time"] += end_reid - start_reid

            if len(bbox_list) > 0:
                """Detections"""
                # tensor -> array ->list ->array
                all_detdata = []
                for (xywh, score, f) in zip(bbox_list, score_list, id_feature):
                    det_data = {
                        "bbox_tlwh": list(xywh),
                        "conf": score,
                        "emb": f.tolist(),
                    }
                    all_detdata.append(det_data)
            else:
                all_detdata = []

            "调用MOT匹配"
            start_match = time.time()
            tracked_stracks = tracktor.update(seqkey, all_detdata)
            end_match = time.time()
            matchmes_dict[seqkey]["match_time"] += end_match - start_match
            online_tlwhs = []
            online_ids = []
            if tracked_stracks:
                # all track, not only is_activated
                online_targets = [track for track in tracked_stracks if track]
                # online_targets = [track for track in tracked_stracks if track.is_activated]
                matchmes_dict[seqkey]["tracked_nums"] += len(online_targets)
                for t in online_targets:
                    # note : 检测框换成跟踪框
                    tlwh = t.det_tlwh
                    # tlwh = t.tlwh
                    tid = t.track_id
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)

            trackres_dict[seqkey].append((matchmes_dict[seqkey]["frame_id"] + 1, online_tlwhs, online_ids))
            if args.draw_save:
                save_singleimg(output_path, args.output_name,writer_dict,seqkey,ori_img,matchmes_dict[seqkey]["frame_id"],
                                online_tlwhs,online_ids,args.draw_det)

    write_allres(output_path,args.output_name,trackres_dict)
    print_summary(matchmes_dict)
    release_matchmes(matchmes_dict)
    release_trackres(trackres_dict)
    if args.draw_save:
        release_writer(writer_dict)


def run_multivideos(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.sgpu_infer
    print("run tracking...")
    allsub_imgs, max_len = load_multiaicity(osp.join(args.input_root,args.scence_id),final_dir = "imgs", select_floder=(args.cam_id))
    pro_len = max_len if max_len < args.pro_len else args.pro_len
    print("pro frame nums is %d." % pro_len)
    eval_multiseq(args, allsub_imgs, pro_len)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #gpu
    parser.add_argument("--sgpu_infer", type=str, default="0", help="single gpu for model infer")

    #path
    parser.add_argument('--input_root', type=str, default="datasets/AIC21_Track3_MTMC_Tracking/validation",help="path to the input image")
    parser.add_argument("--det_result", type=str, default='resultpipeline/expand', help="offline detection result")
    parser.add_argument("--output_root", type=str, default="resultpipeline/mot", help="expected output root path")
    parser.add_argument('--scence_id', type=str, default="S02",help="scence id")
    parser.add_argument("--cam_id", type=str, default='c006', help="cam id")
    parser.add_argument("--output_name", type=str, default="self3", help="expected output name")

    #det
    parser.add_argument("--det_type", type=str, default='mask_rcnn_R', help="offline detection type")
    parser.add_argument("--det_conf", type=float,  default=0.5, help="detection and track confidence")
    parser.add_argument("--det_size_w", type=int, default=25, help="detection width")
    parser.add_argument("--det_size_h", type=int, default=25, help="detection height")
    parser.add_argument("--scale_hw", type=float, default=100.0, help="detection height")

    #track
    parser.add_argument("--emb_type", type=str, default='fastnet', choices=['normalnet','aicitynet','fastnet'], help="emb_type")  
    parser.add_argument("--emb_model", type=str, default='weights/embedding/model_best.pth', help="reid_model")
    parser.add_argument('--track_type',type=str, default='dense', choices=['dense','embedding','distance'], help='track_type')
    parser.add_argument('--feat_alpha',type=float, default=0.9 , help='feat_alpha for smooth_feat,(1-feat_alpha) for curr_feat')
    parser.add_argument('--embedding_thre',type=float, default=0.7 , help='embedding_thre')
    parser.add_argument('--iou_thre1',type=float, default=0.1 , help='iou_thre1')
    parser.add_argument('--iou_thre2',type=float, default=0.5 , help='iou_thre2')

    #visual
    parser.add_argument("--draw_save", action='store_true', help="draw and save(img and video) track result?")
    parser.add_argument("--draw_det", action='store_true', help="draw and scr det result?")
    parser.add_argument("--pro_len", type=int, default=10000, help="action when pro_len < max_len in all sequences") 

    args = parser.parse_args()
    run_multivideos(args)