import argparse
import os
import os.path as osp
import numpy as np
import tqdm
import cv2
import time
import json
from scipy.spatial.distance import cdist
from multiprocessing import Pool
import multiprocessing as mul
import shutil

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from evaluation.evaldet.load import LoadMOTGT,FRestructMot,FusionIGA

from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_test_loader

from pipeline.embedding.fastnet.predictor import FeatureExtractionDemo
from pipeline.track.track_utils import matching
from run_movement import tracklets4cftid,analyse_outmovement


def tracklets4cid(tid_redict, tid_outmovement, max_length):
    all_tracklets = {}
    for track_id, track_megs in tid_redict.items():
        if track_id not in tid_outmovement.keys():
            continue
        cam, camid,frame_id, tlwh = track_megs[0]
        if camid not in all_tracklets.keys():
            all_tracklets[camid] = {}  
        track_length = len(track_megs)
        inter = 0
        if int(track_length/max_length)>1:
            inter = int(track_length/max_length) - 1 
        
        all_tracklets[camid][track_id] = {'frame_id':list(),'feat':[],'length':track_length,
                                            'inter_record':inter,'inter':inter}
    return all_tracklets


def tracklets4tid(tid_redict):
    all_tracklets = {}
    for track_id, track_megs in tid_redict.items():
        cam, camid,frame_id, tlwh = track_megs[0] 
        all_tracklets[track_id] = {'cam_id':camid,'frame_id':[],'feat':[]}
    return all_tracklets


def extract_frame_img(aicitytest_path,cam,img_path,frame_id,frame_list):
    img = cv2.imread(img_path)
    for frame_mes in frame_list:
        _, _, track_id, tlwh = frame_mes
        x,y,w,h = tlwh
        clip = img[y:(y+h),x:(x+w)]
        im_name = '{:05d}'.format(track_id)+"_"+cam+"_"+str(frame_id).zfill(4)+".jpg"
        image_test_path = osp.join(aicitytest_path,"image_test")
        track_path =  osp.join(image_test_path,cam,'{:05d}'.format(track_id))
        cv2.imwrite(osp.join(track_path,im_name),clip)  


def extract_feat(args):
    cfg = get_cfg()
    cfg.merge_from_file('configs/inference/config.yaml')
    cfg.MODEL.WEIGHTS = args.emb_model
    test_loader, num_query = build_reid_test_loader(cfg, 'Aicity'+args.scence_id)
    demo = FeatureExtractionDemo(cfg, parallel=True)
    feats = []
    tids = []
    camids = []
    img_paths = []
    for (img_path ,feat, tid, camid) in tqdm.tqdm(demo.run_on_loader(test_loader), total=len(test_loader)):
        feats.append(feat)
        tids.extend(tid)
        camids.extend(camid)
        img_paths.extend(img_path)
    feats = torch.cat(feats, dim=0).numpy()
    return feats, tids, camids, img_paths


def embedding_distance(a_features, b_features, metric='cosine'):
    """
    :param tracklets_a: dict{'feat':[] ,'frame id' []}
    :param tracklets_b: dict{'feat':[] ,'frame id' []}
    :param metric:
    :return: cost_matrix np.ndarray
    """
    cost_matrix = np.zeros((len(a_features), len(b_features)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix
    cost_matrix = np.maximum(0.0, cdist(a_features, b_features, metric))  # Nomalized features
    return cost_matrix


def match_tracklets(all_tracklets,computed,dthre=0.8):
    matched = []
    paired = []
    # i_camid in all cams
    for i_camid, i_cam_tracklets in all_tracklets.items():
        # i_cam_tracklet in all i_camid tracklets
        for i_tid, i_cam_tracklet in i_cam_tracklets.items():
            if len(i_cam_tracklet["feat"]) != 0:
                i_min_distance = float("inf")
                i_min_jtid = -1
                # j_camid in all cams
                for j_camid, j_cam_tracklets in all_tracklets.items():
                    if i_camid != j_camid:
                        # j_cam_tracklet in all j_camid tracklets
                        for j_tid, j_cam_tracklet in j_cam_tracklets.items():
                            if j_tid > i_tid:
                                if len(j_cam_tracklet["feat"]) != 0:
                                    curpair = (i_tid, j_tid)
                                    mean_distance = computed[curpair]
                                    if mean_distance<dthre:
                                        if mean_distance < i_min_distance:
                                            i_min_distance = mean_distance
                                            i_min_jtid = j_tid
                if i_min_jtid != -1 and i_min_jtid not in paired:
                    matched.append({i_tid,i_min_jtid})
                    paired.append(i_min_jtid)
    return matched


def make_needcomputed(all_tracklets):
    trackpair_list = []
    feat_list = []
    # i_camid in all cams
    # all_tracklets_list = sorted(all_tracklets.items(), key = lambda kv:kv[0])
    for i_camid,i_cam_tracklets in all_tracklets.items():
        # i_cam_tracklet in all i_camid tracklets
        for i_tid, i_cam_tracklet in i_cam_tracklets.items():
            # j_camid in all cams
            if len(i_cam_tracklet["feat"]) != 0:
                for j_camid, j_cam_tracklets in all_tracklets.items():
                    if i_camid != j_camid:
                        # j_cam_tracklet in all j_camid tracklets
                        for j_tid, j_cam_tracklet in j_cam_tracklets.items():
                            if j_tid > i_tid:
                                if len(j_cam_tracklet["feat"]) != 0:
                                    # 先小后大的元组
                                    trackpair_list.append((i_tid, j_tid))
                                    # trackpair_list.append({i_tid, j_tid})
                                    feat_list.append([i_cam_tracklet["feat"], j_cam_tracklet["feat"]]) 
    return trackpair_list,feat_list


def analyse_matched(matched):
    for i_index,i_pair in enumerate(matched):
        for j_index,j_pair in enumerate(matched):
            if i_index >= j_index:
                continue
            intersection = i_pair & j_pair
            if intersection:
                matched[i_index] = i_pair | j_pair
                matched.pop(j_index)
                return False
    return True


#〈camera_id〉 〈obj_id〉 〈frame_id〉 〈xmin〉 〈ymin〉 〈width〉 〈height〉 〈xworld〉 〈yworld〉
def write_tracklets(mtmc_root, scence_id, output_name, mtmc_dict, tid_redict):
    write_path = osp.join(mtmc_root,scence_id)
    if not osp.exists(write_path):
        os.makedirs(write_path)
    with open(osp.join(write_path,output_name+'.txt'), 'w') as write_mtmc:
        for tid, tracklet in tid_redict.items():
            if tid not in mtmc_dict.keys():
                continue
            for track_meg in tracklet:
                cam, camid, frame_id, tlwh = track_meg
                x,y,w,h = tlwh
                wr = '{0} {1} {2} {3} {4} {5} {6} 0 0\n'.format(str(camid),mtmc_dict[tid],frame_id,x,y,w,h)
                write_mtmc.writelines(wr)
    write_mtmc.close()


def write_computed(mtmc_root, scence_id, output_name, computed):
    write_path = osp.join(mtmc_root,scence_id,output_name)
    if not osp.exists(write_path):
        os.makedirs(write_path)
    with open(osp.join(write_path,'computed.json'), 'w') as write_computed:
        computed = list_sort(computed,'dist')
        write_computed(json.dumps(dict(computed)))


def list_sort(random, *args):
    for i in args:
        random = sorted(random, key = lambda kv:kv[i])
    return random


def make_distance(a_features,b_features):
    cost_matrix = embedding_distance(a_features, b_features)
    mean_distance = float(np.mean(cost_matrix))
    # curpair['dist'] = mean_distance
    return mean_distance


def make_distance_api(args_list):
    dict_share = args_list[0]
    track_pair = args_list[1]
    feat_list = args_list[2]
    mean_distance = make_distance(feat_list[0],feat_list[1])
    dict_share[track_pair] = mean_distance


def extract_img(data_path,scence_id,aicitytest_path,cfid_redict):
    for cam ,cam_mes in cfid_redict.items():
        for frame_id ,frame_list in cam_mes.items():
            scence_path = osp.join(data_path,scence_id)
            imgs_path = osp.join(scence_path,cam,'imgs')
            img_path = osp.join(imgs_path, '{:04d}.jpg'.format(frame_id))
            img = cv2.imread(img_path)
            for frame_mes in frame_list:
                _, _, track_id, tlwh = frame_mes
                x,y,w,h = tlwh
                clip = img[y:(y+h),x:(x+w)]
                im_name = '{:05d}'.format(track_id)+"_"+cam+"_"+str(frame_id).zfill(4)+".jpg"
                image_test_path = osp.join(aicitytest_path,"image_test")
                track_path =  osp.join(image_test_path,'{:05d}'.format(track_id))
                if not osp.exists(track_path):
                    os.makedirs(track_path)
                cv2.imwrite(osp.join(track_path,im_name),clip)  
                if not osp.exists(osp.join(aicitytest_path,"image_query")):
                    os.makedirs(osp.join(aicitytest_path,"image_query"))
                if not osp.exists(osp.join(aicitytest_path,"image_train")):
                    os.makedirs(osp.join(aicitytest_path,"image_train"))


def extract_im_api(args_list):
    extract_frame_img(args_list[0],args_list[1],args_list[2],args_list[3],args_list[4])


def extract_img_mlp(data_path,scence_id,n_job,tid_redict,cfid_redict,tid_outmovement):
    print("runing extract_img .....")
    aicitytest_path = osp.join('datasets','aicity'+scence_id)
    image_train_path = osp.join(aicitytest_path,"image_train")
    image_query_path = osp.join(aicitytest_path,"image_query")
    image_test_path = osp.join(aicitytest_path,"image_test")
    if not osp.exists(image_train_path):
        os.makedirs(image_train_path)
    if not osp.exists(image_query_path):
        os.makedirs(image_query_path)
    if not osp.exists(image_test_path):
        os.makedirs(image_test_path)

    for tid in tid_redict:
        if tid not in tid_outmovement.keys():
            continue
        cam, _, _, _ = tid_redict[tid][0]
        track_path = osp.join(image_test_path,cam,'{:05d}'.format(tid))
        if not osp.exists(track_path):
            os.makedirs(track_path)

    extract_im_args = []
    for cam ,cam_mes in cfid_redict.items():
        for frame_id ,frame_list in cam_mes.items():
            scence_path = osp.join(args.data_path,scence_id)
            imgs_path = osp.join(scence_path,cam,'imgs')
            img_path = osp.join(imgs_path, '{:04d}.jpg'.format(frame_id))
            extract_im_args.append([aicitytest_path,cam,img_path,frame_id,frame_list])
    extractimg_pool = Pool(n_job)
    extractimg_pool.map(extract_im_api, extract_im_args)
    extractimg_pool.close()
    print("end extract_img .....")


#自西向东
def match_w2e(all_tracklets,computed,tid_redict,tid_outmovement,dthre):
    matched = []
    #inter_time = [[362,605],[272,323],[475,542],[287,512],[262,585]]
    inter_time = [[262,905],[172,623],[375,842],[187,812],[162,885]]
    sorted_cams = [46,45,44,43,42,41]
    for index in range(len(sorted_cams)):
        if index == len(sorted_cams)-1:
            #到达最后一个相机
            break
        cur_camid,next_camid = sorted_cams[index],sorted_cams[index+1]
        cur_cam_tracklets,next_cam_tracklets = all_tracklets[cur_camid],all_tracklets[next_camid]

        curtid_filter = []
        curtracklet_filter = []
        for cur_tid, cur_cam_tracklet in cur_cam_tracklets.items():
            outmovement = tid_outmovement[cur_tid]
            if outmovement in [11,3,7]:
                curtid_filter.append(cur_tid)
                curtracklet_filter.append(cur_cam_tracklet)

        nexttid_filter = []
        nexttracklet_filter = []
        for next_tid, next_cam_tracklet in next_cam_tracklets.items():
            outmovement = tid_outmovement[next_tid]
            if outmovement in [10,11,12]:
                nexttid_filter.append(next_tid)
                nexttracklet_filter.append(next_cam_tracklet)

        #匹配curtid_filter和nexttid_filter
        cost_matrix = np.zeros((len(curtid_filter), len(nexttid_filter)), dtype=np.float)
        for cur_index, cur_tid in enumerate(curtid_filter):
            cur_tracklet = tid_redict[cur_tid]
            cur_frame_list = [i[2] for  i in cur_tracklet]
            #最后消失的时间
            last_frame = max(cur_frame_list)
            for next_index, next_tid in enumerate(nexttid_filter):
                next_tracklet = tid_redict[next_tid]
                next_frame_list = [i[2] for  i in next_tracklet]
                #第一次出现的时间
                first_frame = min(next_frame_list)
                #时间逻辑，如果最后消失的时间大于第一次出现的时间
                if last_frame> first_frame:
                    cost_matrix[cur_index, next_index] = np.inf
                    continue
                if first_frame - last_frame < inter_time[index][0]:
                    cost_matrix[cur_index, next_index] = np.inf
                    continue
                if first_frame - last_frame > inter_time[index][1]:
                    cost_matrix[cur_index, next_index] = np.inf
                    continue
                if cur_tid<next_tid:
                    curpair = (cur_tid, next_tid)
                else:
                    curpair = (next_tid, cur_tid)
                mean_distance = computed[curpair]
                cost_matrix[cur_index, next_index] = mean_distance
        matches, u_cur, u_next = matching.linear_assignment(cost_matrix, thresh=dthre)
        for cur_index, next_index in matches:
            cur_tid = curtid_filter[cur_index]
            next_tid = nexttid_filter[next_index]
            matched.append({cur_tid,next_tid})
    return matched

#自东向西
def match_e2w(all_tracklets,computed,tid_redict,tid_outmovement,dthre):
    matched = []
    #inter_time = [[299,1243],[176,694],[362,450],[120,254],[320,400]]
    inter_time = [[199,1543],[76,994],[262,750],[20,554],[220,700]]
    sorted_cams = [41,42,43,44,45,46]
    for index in range(len(sorted_cams)):
        if index == len(sorted_cams)-1:
            #到达最后一个相机
            break
        cur_camid,next_camid = sorted_cams[index],sorted_cams[index+1]
        cur_cam_tracklets,next_cam_tracklets = all_tracklets[cur_camid],all_tracklets[next_camid]

        curtid_filter = []
        curtracklet_filter = []
        for cur_tid, cur_cam_tracklet in cur_cam_tracklets.items():
            outmovement = tid_outmovement[cur_tid]
            if outmovement in [1,9,5]:
                curtid_filter.append(cur_tid)
                curtracklet_filter.append(cur_cam_tracklet)


        nexttid_filter = []
        nexttracklet_filter = []
        for next_tid, next_cam_tracklet in next_cam_tracklets.items():
            outmovement = tid_outmovement[next_tid]
            if outmovement in [5,4,6]:
                nexttid_filter.append(next_tid)
                nexttracklet_filter.append(next_cam_tracklet)


        #匹配curtid_filter和nexttid_filter
        cost_matrix = np.zeros((len(curtid_filter), len(nexttid_filter)), dtype=np.float)
        for cur_index, cur_tid in enumerate(curtid_filter):
            cur_tracklet = tid_redict[cur_tid]
            cur_frame_list = [i[2] for  i in cur_tracklet]
            #最后消失的时间
            last_frame = max(cur_frame_list)
            for next_index, next_tid in enumerate(nexttid_filter):
                next_tracklet = tid_redict[next_tid]
                next_frame_list = [i[2] for  i in next_tracklet]
                #第一次出现的时间
                first_frame = min(next_frame_list)
                #时间逻辑，如果最后消失的时间大于第一次出现的时间
                if last_frame > first_frame:
                    cost_matrix[cur_index, next_index] = np.inf
                    continue
                if first_frame - last_frame < inter_time[index][0]:
                    cost_matrix[cur_index, next_index] = np.inf
                    continue
                if first_frame - last_frame > inter_time[index][1]:
                    cost_matrix[cur_index, next_index] = np.inf
                    continue
                if cur_tid < next_tid:
                    curpair = (cur_tid, next_tid)
                else:
                    curpair = (next_tid, cur_tid)
                mean_distance = computed[curpair]
                cost_matrix[cur_index, next_index] = mean_distance
        matches, u_cur, u_next = matching.linear_assignment(cost_matrix, thresh=dthre)
        for cur_index, next_index in matches:
            cur_tid = curtid_filter[cur_index]
            next_tid = nexttid_filter[next_index]
            matched.append({cur_tid,next_tid})
    return matched


def merge_match(match_we):
    mtmc_dict = {}
    new_tid = 1
    for matched in match_we:
        while True:
            flag = analyse_matched(matched)
            if flag:
                break
        for pair in matched:
            pair = sorted(list(pair))
            for key in pair:
                mtmc_dict[key]=new_tid
            new_tid+=1
    return mtmc_dict


def match_tracklets_spacetime(all_tracklets,computed,tid_redict,tid_outmovement,dthre):
    matched_w2e = match_w2e(all_tracklets,computed,tid_redict,tid_outmovement,dthre)
    matched_e2w = match_e2w(all_tracklets,computed,tid_redict,tid_outmovement,dthre)
    return matched_w2e, matched_e2w


def run_multicams(args):
    tid_redict, cfid_redict = tracklets4cftid(args.data_path,args.filter_root,args.scence_id,args.filter_type)
    tid_outmovement = analyse_outmovement(tid_redict)

    args.extract_img = True
    if args.extract_img:
        extract_img_mlp(args.data_path,args.scence_id,args.n_job,tid_redict,cfid_redict,tid_outmovement)

    print("runing extract_feat .....")
    extract_feat_time = time.time()
    feats, tids, camids, img_paths = extract_feat(args)
    print('extract_feat time:', time.time() - extract_feat_time)

    all_tracklets = tracklets4cid(tid_redict,tid_outmovement,args.max_length)
    for camid, tid, img_path, feat in zip(camids, tids, img_paths, feats):
        if all_tracklets[int(camid)][int(tid)]['inter'] > 0:
            all_tracklets[int(camid)][int(tid)]['inter'] -= 1
            continue
        frame_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
        all_tracklets[int(camid)][int(tid)]['frame_id'].append(int(frame_id)) 
        all_tracklets[int(camid)][int(tid)]['feat'].append(feat) 
        all_tracklets[int(camid)][int(tid)]['inter'] = all_tracklets[int(camid)][int(tid)]['inter_record']
    
    print("runing make_needcomputed .....")
    make_needcomputed_time = time.time()
    need_trackpair, need_featpair = make_needcomputed(all_tracklets)
    print('make_needcomputed time:', time.time() - make_needcomputed_time)
    
    print("runing make_distance .....")
    make_distance_time = time.time()
    mgr = mul.Manager()
    dict_share = mgr.dict()
    make_distance_args = []
    for trackpair, featpair in zip(need_trackpair, need_featpair):
        make_distance_args.append([dict_share,trackpair,featpair])
    makedistance_pool = Pool(args.n_job)
    makedistance_pool.map(make_distance_api, make_distance_args)
    makedistance_pool.close()
    print('make_distance time:', time.time() - make_distance_time)

    computed2dict_time = time.time()
    computed = dict(dict_share)
    print('computed2dict time:', time.time() - computed2dict_time )
    
    #matching all tracklets in space-time
    matched_w2e, matched_e2w = match_tracklets_spacetime(all_tracklets,computed,tid_redict,
                                    tid_outmovement,dthre=args.matching_thres)
    mtmc_dict = merge_match((matched_w2e,matched_e2w))
    write_tracklets(args.mtmc_root, args.scence_id, args.mtmc_type, mtmc_dict, tid_redict)

    print("write result done")
    if args.draw_save:
        for tid, tracklet in tid_redict.items():
            cam, camid, _, _ = tracklet[0]
            tid_srcfloder = osp.join('datasets', 'aicity{}'.format(args.scence_id),"image_test", cam, '{:05d}'.format(tid))
            if tid not in tid_outmovement.keys():
                continue
            if tid not in mtmc_dict.keys():
                tid_dstfloder = osp.join(args.mtmc_root, args.scence_id, args.mtmc_type, 'unmatching')
            else:
                savetid = mtmc_dict[tid]
                tid_dstfloder = osp.join(args.mtmc_root, args.scence_id, args.mtmc_type, 'matched','{:05d}'.format(savetid), cam)
            if not osp.exists(tid_dstfloder):
                os.makedirs(tid_dstfloder)
            shutil.move(tid_srcfloder,tid_dstfloder)


def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str, default="datasets/AIC21_Track3_MTMC_Tracking/test",
                        help='path to the aicity 2021 track 3 folders')
    parser.add_argument("--filter_root", type=str, default="resultpipeline/filter", help="expected output root path")
    parser.add_argument("--mtmc_root", type=str, default="resultpipeline/mtmc", help="expected output root path")
    parser.add_argument("--filter_type", type=str, default="self19", help="expected input filter name")
    parser.add_argument("--mtmc_type", type=str, default="self50", help="expected output mtmc name")
    parser.add_argument('--scence_id', type=str, default="S06",help="scence id")
    parser.add_argument("--emb_model", type=str, default='weights/embedding/model_best.pth', help="reid_model")
    parser.add_argument("--n_job", type=int, default=40, help="mulit process nums")
    parser.add_argument("--matching_thres", type=float, default=0.8, help="match thres")
    parser.add_argument("--max_length", type=int, default=20, help="match thres")
    parser.add_argument("--extract_img", action='store_true', help="need extract_img again")
    parser.add_argument("--draw_save", action='store_true', help="draw and save mtmc result?")
    return parser


if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    run_multicams(args)
