# 单个相机内进行过滤操作，过滤后的多目标跟踪结果，都要参与mtmc的匹配，不进行二次过滤
import argparse
import os
import os.path as osp
import numpy as np
import tqdm
import cv2
import json

from run_movement import tracklets4cftid,FRestructTrack
from evaluation.evaldet.load import FRestructMot,LoadMOTGT
from pipeline.utils.draw import plot_tracking


def analyse_position(track_list, xvar_thres, yvar_thres, logic):
    x_list, y_list, w_list, h_list = [],[],[],[]
    for track in track_list:
        tlwh , frame_id, _ = track
        x,y,w,h = tlwh
        x_list.append(x)
        y_list.append(y)
        w_list.append(w)
        h_list.append(h)

    x_var,y_var,w_var,h_var = \
        np.var(x_list),np.var(y_list),np.var(w_list),np.var(h_list)
    if logic == "and":
        if x_var < xvar_thres and y_var < yvar_thres:
            return True
    elif logic == "or":
        if x_var < xvar_thres or y_var < yvar_thres:
            return True
    else:
        print('error logic type')
    return False


def run_filter(args):
    scence_path = osp.join(args.mot_root, args.scence_id)

    cams = os.listdir(scence_path)
    for cam in sorted(cams):
        tracklet_path = osp.join(scence_path,cam,'mot')
        tracklets = os.listdir(tracklet_path)
        tracklet_file = [i for i in tracklets if args.file_type in i][0] 

        out_path = osp.join(args.filter_root, args.scence_id, cam, 'filter')
        if not osp.exists(out_path):
            os.makedirs(out_path)
        with open(osp.join(out_path,tracklet_file), "w") as writer:
            with open(osp.join(tracklet_path,tracklet_file), "r") as reader:
                lines = reader.readlines() 
            valid_redict = FRestructTrack(lines)
            for track_id,track_list in valid_redict.items():
                # 过滤掉track长度小于3的跟踪id
                if len(track_list)<5:
                    continue
                # 过滤静止不动的车
                if analyse_position(track_list,args.xvar_thres, args.yvar_thres ,args.logic):
                    continue
                for track in track_list:
                    tlwh , frame_id, _ = track
                    if args.scence_id == 'S06' and frame_id>2000:
                        continue
                    x,y,w,h = tlwh
                    line = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'.format(frame=frame_id, id=track_id, x1=x, y1=y, w=w, h=h)
                    writer.write(line)
        writer.close()
        reader.close()

    if args.draw_save:
        tid_redict, cfid_redict = tracklets4cftid(args.data_path, args.filter_root, args.scence_id, args.file_type)
        for cam, cam_megs in cfid_redict.items():
            draw_path = osp.join(args.filter_root,args.scence_id,cam,args.file_type,'imgs')
            print(draw_path)
            if not osp.exists(draw_path):
                os.makedirs(draw_path)
            imgs_path = osp.join(args.data_path,args.scence_id,cam,'imgs')
            for frame_id,frame_megs in cam_megs.items():
                img_path = osp.join(imgs_path, '{:04d}.jpg'.format(frame_id))
                tlwh_list=[]
                track_id_list=[]
                for frame_meg in frame_megs:
                    _,_,track_id,tlwh = frame_meg
                    tlwh_list.append(tlwh)
                    track_id_list.append(track_id)
                src_img = cv2.imread(img_path)
                draw_img = plot_tracking(src_img, tlwh_list, track_id_list, frame_id=frame_id, fps=10)
                cv2.imwrite(osp.join(draw_path,'{:04d}.jpg'.format(frame_id)), draw_img)


def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str, default="datasets/AIC21_Track3_MTMC_Tracking/test",
                        help='path to the aicity 2021 track 3 folders')
    parser.add_argument("--mot_root", type=str, default="resultpipeline/mot", help="expected output root path")
    parser.add_argument("--filter_root", type=str, default="resultpipeline/filter", help="expected output root path")
    parser.add_argument('--scence_id', type=str, default="S06",help="scence id")
    parser.add_argument("--file_type", type=str, default="self42", help="expected output name")
    parser.add_argument("--xvar_thres", type=int, default=20, help="xvar_thres")
    parser.add_argument("--yvar_thres", type=int, default=20, help="yvar_thres")
    parser.add_argument("--logic", type=str, default="and", help=" and  or")
    parser.add_argument("--draw_save", action='store_true', help="draw and save(img and video) track result?")
    return parser


if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    run_filter(args)
